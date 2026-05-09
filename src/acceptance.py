import torch
import torch.nn.functional as F


def get_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    return F.softmax(logits / temperature, dim=-1)


def acceptance_prob(p_probs: torch.Tensor, q_probs: torch.Tensor, token_id: int) -> float:
    p = p_probs[token_id].item()
    q = q_probs[token_id].item()
    return min(1.0, p / (q + 1e-10))


def sample_residual(p_probs: torch.Tensor, q_probs: torch.Tensor) -> int:
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    total = residual.sum()
    if total < 1e-8:
        return torch.multinomial(p_probs, num_samples=1).item()
    residual = residual / total
    return torch.multinomial(residual, num_samples=1).item()
