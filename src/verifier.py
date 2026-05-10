import torch
from utils import get_device, load_model


class VerifierModel:
    def __init__(self, model_name: str, device):
        self.model, self.tokenizer = load_model(model_name, device)
        self.device = device

    def score(self, prefix_ids: torch.Tensor, draft_tokens: torch.Tensor, temperature: float = 0.0):
        """
        Score K draft tokens in a single parallel forward pass.

        Args:
            prefix_ids: (1, L) the prompt token ids
            draft_tokens: (K,) the K candidate tokens proposed by the draft model

        Returns:
            verifier_probs: (K+1, vocab_size)
                rows 0..K-1: prob distribution at each draft position, used to accept or reject
                row K:       bonus token distribution, sampled from if all K are accepted
        """
        K = draft_tokens.shape[0]
        L = prefix_ids.shape[1]

        draft_ids = draft_tokens.view(1, -1)
        full_input = torch.cat([prefix_ids, draft_ids], dim=1)  # (1, L+K)

        with torch.no_grad():
            out = self.model(full_input, use_cache=False)

        logits = out.logits[0]  # (L+K, vocab_size)

        # logit at position i predicts the token at position i+1
        # so logits[L-1] predicts t1, logits[L] predicts t2, ..., logits[L+K-1] predicts the bonus
        relevant = logits[L - 1 : L + K, :]  # (K+1, vocab_size)

        return _batch_get_probs(relevant, temperature)


def _batch_get_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        argmax_idx = torch.argmax(logits, dim=-1)
        probs[torch.arange(logits.shape[0]), argmax_idx] = 1.0
        return probs
    return torch.softmax(logits / temperature, dim=-1)


if __name__ == "__main__":
    from draft import DraftModel

    device = get_device()
    print(f"device: {device}\n")

    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)

    prompt = "The transformer architecture"
    input_ids = draft.tokenizer.encode(prompt, return_tensors="pt").to(device)

    k = 4
    draft_tokens, draft_probs = draft.propose(input_ids, k=k, temperature=0.0)
    verifier_probs = verifier.score(input_ids, draft_tokens, temperature=0.0)

    print(f"draft proposed {k} tokens, verifier scored them in 1 forward pass:\n")
    for i in range(k):
        token_id = draft_tokens[i].item()
        token_str = draft.tokenizer.decode([token_id])
        q = draft_probs[i, token_id].item()
        p = verifier_probs[i, token_id].item()
        verdict = "verifier agrees" if p > 0.5 else "verifier disagrees"
        print(f"  [{i}] '{token_str}' (id={token_id})  q={q:.4f}  p={p:.4f}  -> {verdict}")

    bonus_id = torch.argmax(verifier_probs[k]).item()
    bonus_str = verifier.tokenizer.decode([bonus_id])
    bonus_p = verifier_probs[k, bonus_id].item()
    print(f"\n  bonus: '{bonus_str}' (id={bonus_id}, p={bonus_p:.4f})")

    print(f"\nverifier_probs shape: {verifier_probs.shape}")
