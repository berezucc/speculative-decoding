import argparse
import torch
from utils import get_device, set_seed
from draft import DraftModel
from verifier import VerifierModel
from acceptance import acceptance_prob, sample_residual


def speculative_generate(
    draft: DraftModel,
    verifier: VerifierModel,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    device=None,
):
    """
    Speculative decoding loop.
    Returns:
        generated_tokens: list of token ids (length == n_tokens)
        stats: dict with total_proposed, total_accepted, acceptance_rate
    """
    initial_ids = verifier.tokenizer.encode(prompt, return_tensors="pt").to(device)
    L0 = initial_ids.shape[1]
    current_ids = initial_ids

    total_proposed = 0
    total_accepted = 0

    while current_ids.shape[1] - L0 < n_tokens:
        # 1. draft proposes K tokens
        draft_tokens, draft_probs = draft.propose(current_ids, k=k, temperature=temperature)

        # 2. verifier scores all K + 1 bonus in one parallel pass
        verifier_probs = verifier.score(current_ids, draft_tokens, temperature=temperature)

        # 3. walk through draft tokens, accept or reject each
        new_tokens = []
        rejected = False
        for i in range(k):
            token_id = draft_tokens[i].item()
            p = verifier_probs[i]
            q = draft_probs[i]

            alpha = acceptance_prob(p, q, token_id)
            r = torch.rand(1).item()

            if r < alpha:
                new_tokens.append(token_id)
            else:
                # reject and sample correction from residual, discard rest
                correction = sample_residual(p, q)
                new_tokens.append(correction)
                rejected = True
                break

        n_accepted_this_iter = len(new_tokens) - (1 if rejected else 0)

        # 4. if all K accepted, sample bonus token from verifier
        if not rejected:
            bonus_probs = verifier_probs[k]
            if temperature == 0.0:
                bonus = torch.argmax(bonus_probs).item()
            else:
                bonus = torch.multinomial(bonus_probs, num_samples=1).item()
            new_tokens.append(bonus)

        # append new tokens to running sequence
        new_tokens_tensor = torch.tensor([new_tokens], dtype=current_ids.dtype, device=device)
        current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)

        total_proposed += k
        total_accepted += n_accepted_this_iter

    # truncate to exactly n_tokens
    generated = current_ids[0, L0 : L0 + n_tokens].tolist()

    stats = {
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0.0,
    }
    return generated, stats


def test_correctness(draft, verifier, device):
    """At temp=0, speculative output must match verifier-only greedy exactly."""
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
        print(f"PASS: speculative output matches verifier-only greedy ({n_tokens} tokens)")
        print(f"      acceptance rate: {stats['acceptance_rate']:.2%}")
    else:
        print("FAIL: outputs differ")
        print(f"  greedy: {greedy_tokens}")
        print(f"  spec:   {spec_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The transformer architecture")
    parser.add_argument("--n_tokens", type=int, default=50)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test", action="store_true", help="run correctness test against greedy")
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
