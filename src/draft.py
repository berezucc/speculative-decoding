import torch
from utils import get_device, load_model


class DraftModel:
    def __init__(self, model_name: str, device):
        self.model, self.tokenizer = load_model(model_name, device)
        self.device = device

    def propose(self, input_ids: torch.Tensor, k: int, temperature: float = 0.0):
        """
        Run the draft model autoregressively for k steps.
        Returns:
            draft_tokens: (k,) tensor of proposed token ids
            draft_probs:  (k, vocab_size) tensor of full probability distributions
        """
        past_key_values = None
        draft_tokens = []
        draft_probs = []

        current_input = input_ids

        with torch.no_grad():
            for _ in range(k):
                out = self.model(current_input, past_key_values=past_key_values, use_cache=True)
                logits = out.logits[0, -1, :]        # (vocab_size,)
                past_key_values = out.past_key_values

                probs = _get_probs(logits, temperature)
                next_token = _sample(probs)

                draft_tokens.append(next_token)
                draft_probs.append(probs)

                current_input = next_token.view(1, 1)

        draft_tokens = torch.stack(draft_tokens)       # (k,)
        draft_probs = torch.stack(draft_probs)         # (k, vocab_size)

        return draft_tokens, draft_probs


def _get_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs: torch.Tensor) -> torch.Tensor:
    if probs.max().item() == 1.0:
        return torch.argmax(probs)
    return torch.multinomial(probs, num_samples=1).squeeze()


if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}\n")

    draft = DraftModel("distilgpt2", device)

    prompt = "The transformer architecture"
    input_ids = draft.tokenizer.encode(prompt, return_tensors="pt").to(device)

    k = 4
    draft_tokens, draft_probs = draft.propose(input_ids, k=k, temperature=0.0)

    print(f"proposed {k} tokens:")
    for i, (tok, probs) in enumerate(zip(draft_tokens, draft_probs)):
        token_str = draft.tokenizer.decode([tok.item()])
        prob = probs[tok.item()].item()
        print(f"  [{i}] '{token_str}' (id={tok.item()}, q={prob:.4f})")

    print(f"\ndraft_tokens shape: {draft_tokens.shape}")
    print(f"draft_probs shape:  {draft_probs.shape}")
