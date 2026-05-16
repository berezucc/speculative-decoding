import argparse
import torch
from utils import get_device, set_seed
from draft import DraftModel
from verifier import VerifierModel
from acceptance import acceptance_prob, sample_residual


def _get_probs(logits, temperature):
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def _batch_get_probs(logits, temperature):
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        argmax_idx = torch.argmax(logits, dim=-1)
        probs[torch.arange(logits.shape[0]), argmax_idx] = 1.0
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs):
    if probs.max().item() == 1.0:
        return torch.argmax(probs)
    return torch.multinomial(probs, num_samples=1).squeeze()


def _truncate_kv(past_key_values, target_length):
    if past_key_values is None or target_length is None:
        return past_key_values
    # Newer transformers: Cache object with crop method
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(target_length)
        return past_key_values
    # Legacy: tuple of tuples
    truncated = []
    for layer in past_key_values:
        k, v = layer
        if k.shape[-2] <= target_length:
            truncated.append((k, v))
        else:
            truncated.append((k[..., :target_length, :], v[..., :target_length, :]))
    return tuple(truncated)


def speculative_generate(
    draft_obj: DraftModel,
    verifier_obj: VerifierModel,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    device=None,
):
    """
    Speculative decoding with persistent KV cache across iterations (eager scheme).
    Invariant: at start of each iter, caches cover current_ids[:-1] (all but the last token).
    The last token is fed at the start of each iter as part of the K-step draft proposal
    and the K+1-token verifier scoring pass. Saves 1 forward pass per iter vs the
    saved-logit scheme (no end-of-iter "setup_next" pass needed).
    """
    tokenizer = verifier_obj.tokenizer
    initial_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    L0 = initial_ids.shape[1]
    current_ids = initial_ids.clone()

    # Prefill: feed prompt[:-1] to set up caches covering all but the last prompt token
    with torch.no_grad():
        if L0 > 1:
            prefill_ids = initial_ids[:, :-1]
            verifier_kv = verifier_obj.model(prefill_ids, use_cache=True).past_key_values
            draft_kv = draft_obj.model(prefill_ids, use_cache=True).past_key_values
        else:
            verifier_kv = None
            draft_kv = None

    total_proposed = 0
    total_accepted = 0

    while current_ids.shape[1] - L0 < n_tokens:
        T_old = current_ids.shape[1]
        last_token = current_ids[0, -1].item()

        # 1. Draft proposes K tokens via K forward passes
        # Inlined sampling to avoid the .max().item() sync inside _sample at temp=0.
        draft_tokens = []
        draft_probs_list = []
        prev_input = torch.tensor([[last_token]], device=device)
        is_greedy = temperature == 0.0
        with torch.no_grad():
            for _ in range(k):
                out = draft_obj.model(prev_input, past_key_values=draft_kv, use_cache=True)
                draft_kv = out.past_key_values
                logit = out.logits[0, -1, :]
                if is_greedy:
                    t = torch.argmax(logit)
                    probs = torch.zeros_like(logit)
                    probs[t] = 1.0
                else:
                    probs = torch.softmax(logit / temperature, dim=-1)
                    t = torch.multinomial(probs, num_samples=1).squeeze()
                draft_probs_list.append(probs)
                draft_tokens.append(t)
                prev_input = t.view(1, 1)

        draft_tokens_tensor = torch.stack(draft_tokens)
        draft_probs_tensor = torch.stack(draft_probs_list)

        # 2. Verifier scoring: one pass on [last_token, t1, t2, ..., tK] (K+1 tokens)
        # Yields K+1 logits at positions T_old-1, T_old, ..., T_old+K-1
        # (row 0 scores t1; row K is the bonus prediction)
        with torch.no_grad():
            verifier_input = torch.cat([
                torch.tensor([[last_token]], device=device),
                draft_tokens_tensor.view(1, -1),
            ], dim=1)
            out_v = verifier_obj.model(verifier_input, past_key_values=verifier_kv, use_cache=True)
            verifier_kv = out_v.past_key_values
            new_v_logits = out_v.logits[0]  # (K+1, vocab)
        # verifier_kv now at T_old + K

        verifier_probs = _batch_get_probs(new_v_logits, temperature)  # (K+1, vocab)

        # 3. Accept or reject each draft token
        # Vectorize alpha computation: one .gather() across all K positions, then a single .tolist()
        # to bring alphas to CPU. Pre-roll random numbers on CPU. Total syncs: 2 (was ~4K).
        token_ids_tensor = draft_tokens_tensor.unsqueeze(1)  # (K, 1)
        v_at_tokens = verifier_probs[:k].gather(1, token_ids_tensor).squeeze(1)  # (K,)
        d_at_tokens = draft_probs_tensor.gather(1, token_ids_tensor).squeeze(1)  # (K,)
        alphas_list = torch.clamp(v_at_tokens / (d_at_tokens + 1e-10), max=1.0).tolist()
        token_ids_list = draft_tokens_tensor.tolist()
        r_values = torch.rand(k).tolist()  # CPU rand, no MPS sync

        new_tokens = []
        rejected = False
        for i in range(k):
            if r_values[i] < alphas_list[i]:
                new_tokens.append(token_ids_list[i])
            else:
                correction = sample_residual(verifier_probs[i], draft_probs_tensor[i])
                new_tokens.append(correction)
                rejected = True
                break

        n_accepted = len(new_tokens) - (1 if rejected else 0)

        # 4. Adjust caches to maintain INV: cache covers all but the new last token
        if rejected:
            # new current_ids length = T_old + n_accepted + 1; cache target = T_old + n_accepted
            target_len = T_old + n_accepted
            verifier_kv = _truncate_kv(verifier_kv, target_len)
            draft_kv = _truncate_kv(draft_kv, target_len)
        else:
            # Full accept + bonus. new length = T_old + K + 1; cache target = T_old + K
            bonus = _sample(verifier_probs[k])
            new_tokens.append(bonus.item())
            # verifier_kv already at T_old + K ✓
            # draft_kv at T_old + K - 1, extend by feeding tK
            tK = draft_tokens_tensor[k - 1].view(1, 1)
            with torch.no_grad():
                out_d = draft_obj.model(tK, past_key_values=draft_kv, use_cache=True)
                draft_kv = out_d.past_key_values

        # Append new tokens to current_ids
        new_tokens_tensor = torch.tensor([new_tokens], dtype=current_ids.dtype, device=device)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)

        total_proposed += k
        total_accepted += n_accepted

    generated = current_ids[0, L0 : L0 + n_tokens].tolist()
    stats = {
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0.0,
    }
    return generated, stats


def test_correctness(draft, verifier, device):
    from baseline import greedy_generate

    prompt = "The transformer architecture"
    n_tokens = 30

    set_seed()
    greedy_tokens = greedy_generate(
        verifier.model, verifier.tokenizer, prompt, n_tokens, device, temperature=0.0
    )

    set_seed()
    spec_tokens, stats = speculative_generate(
        draft, verifier, prompt, n_tokens, k=4, temperature=0.0, device=device
    )

    if greedy_tokens == spec_tokens:
        print(f"PASS: cached spec matches verifier-only greedy ({n_tokens} tokens)")
        print(f"      acceptance rate: {stats['acceptance_rate']:.2%}")
    else:
        print("FAIL: outputs differ")
        print(f"  greedy: {greedy_tokens}")
        print(f"  spec:   {spec_tokens}")
        for i, (a, b) in enumerate(zip(greedy_tokens, spec_tokens)):
            if a != b:
                print(f"  first divergence at position {i}: greedy={a}, spec={b}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The transformer architecture")
    parser.add_argument("--n_tokens", type=int, default=50)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}\n")

    print("loading models...")
    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)

    if args.test:
        test_correctness(draft, verifier, device)
    else:
        set_seed()
        tokens, stats = speculative_generate(
            draft, verifier, args.prompt, args.n_tokens,
            k=args.k, temperature=args.temperature, device=device,
        )
        text = verifier.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\nprompt: {args.prompt}")
        print(f"output: {text}")
        print(f"\nproposed: {stats['total_proposed']}  accepted: {stats['total_accepted']}")
        print(f"acceptance rate: {stats['acceptance_rate']:.2%}")
