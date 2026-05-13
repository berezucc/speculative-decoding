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


def speculative_generate_cached(
    draft_obj: DraftModel,
    verifier_obj: VerifierModel,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    device=None,
):
    """
    Speculative decoding with persistent KV cache across iterations.
    Each iteration only processes the new tokens, not the full prefix.
    """
    tokenizer = verifier_obj.tokenizer
    initial_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    L0 = initial_ids.shape[1]
    current_ids = initial_ids.clone()

    # Initial prefill: process entire prompt once, save KV caches and the last logits
    with torch.no_grad():
        out_v = verifier_obj.model(initial_ids, use_cache=True)
        verifier_kv = out_v.past_key_values
        last_v_logit = out_v.logits[0, -1, :].clone()

        out_d = draft_obj.model(initial_ids, use_cache=True)
        draft_kv = out_d.past_key_values
        last_d_logit = out_d.logits[0, -1, :].clone()

    total_proposed = 0
    total_accepted = 0

    while current_ids.shape[1] - L0 < n_tokens:
        T_old = current_ids.shape[1]

        # 1. Draft proposes K tokens
        draft_tokens = []
        draft_probs_list = []

        # First draft uses the saved logit (no forward pass needed)
        q1 = _get_probs(last_d_logit, temperature)
        t1 = _sample(q1)
        draft_tokens.append(t1)
        draft_probs_list.append(q1)

        # Subsequent drafts: K-1 forward passes, each feeding only 1 token
        with torch.no_grad():
            for i in range(1, k):
                input_tok = draft_tokens[-1].view(1, 1)
                out = draft_obj.model(input_tok, past_key_values=draft_kv, use_cache=True)
                draft_kv = out.past_key_values
                logit = out.logits[0, -1, :]
                probs = _get_probs(logit, temperature)
                draft_probs_list.append(probs)
                draft_tokens.append(_sample(probs))

        draft_tokens_tensor = torch.stack(draft_tokens)
        draft_probs_tensor = torch.stack(draft_probs_list)

        # 2. Verifier scoring: feed K draft tokens in one parallel pass
        with torch.no_grad():
            verifier_input = draft_tokens_tensor.view(1, -1)
            out_v = verifier_obj.model(verifier_input, past_key_values=verifier_kv, use_cache=True)
            verifier_kv = out_v.past_key_values
            new_v_logits = out_v.logits[0]  # (k, vocab)

        verifier_logits_all = torch.cat([last_v_logit.unsqueeze(0), new_v_logits], dim=0)  # (k+1, vocab)
        verifier_probs = _batch_get_probs(verifier_logits_all, temperature)

        # 3. Accept or reject each draft token
        new_tokens = []
        rejected = False
        for i in range(k):
            token_id = draft_tokens[i].item()
            p = verifier_probs[i]
            q = draft_probs_tensor[i]

            alpha = acceptance_prob(p, q, token_id)
            r = torch.rand(1).item()

            if r < alpha:
                new_tokens.append(token_id)
            else:
                correction = sample_residual(p, q)
                new_tokens.append(correction)
                rejected = True
                break

        n_accepted = len(new_tokens) - (1 if rejected else 0)
        last_token = new_tokens[-1]

        # 4. Truncate or extend caches based on what was accepted
        if rejected:
            # Cache should cover prefix + n_accepted drafts
            target_len = T_old + n_accepted
            verifier_kv = _truncate_kv(verifier_kv, target_len)
            draft_kv = _truncate_kv(draft_kv, target_len)
        else:
            # All k accepted, sample bonus from verifier_probs[k]
            bonus = _sample(verifier_probs[k])
            new_tokens.append(bonus.item())
            last_token = bonus.item()
            # Verifier cache already at T_old + k (covers all k drafts)
            # Draft cache at T_old + k - 1, extend it by feeding tK
            tK = draft_tokens_tensor[k - 1].view(1, 1)
            with torch.no_grad():
                out_d = draft_obj.model(tK, past_key_values=draft_kv, use_cache=True)
                draft_kv = out_d.past_key_values

        # 5. Feed last_token to both models to set up saved logits for next iter
        last_tok_tensor = torch.tensor([[last_token]], device=device)
        with torch.no_grad():
            out_v = verifier_obj.model(last_tok_tensor, past_key_values=verifier_kv, use_cache=True)
            verifier_kv = out_v.past_key_values
            last_v_logit = out_v.logits[0, -1, :].clone()

            out_d = draft_obj.model(last_tok_tensor, past_key_values=draft_kv, use_cache=True)
            draft_kv = out_d.past_key_values
            last_d_logit = out_d.logits[0, -1, :].clone()

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
    spec_tokens, stats = speculative_generate_cached(
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
        tokens, stats = speculative_generate_cached(
            draft, verifier, args.prompt, args.n_tokens,
            k=args.k, temperature=args.temperature, device=device,
        )
        text = verifier.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\nprompt: {args.prompt}")
        print(f"output: {text}")
        print(f"\nproposed: {stats['total_proposed']}  accepted: {stats['total_accepted']}")
        print(f"acceptance rate: {stats['acceptance_rate']:.2%}")
