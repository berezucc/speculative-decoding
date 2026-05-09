import time
import argparse
import torch
from utils import get_device, load_model, set_seed


def greedy_generate(model, tokenizer, prompt: str, n_tokens: int, device, temperature: float = 0.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None
    generated = []

    with torch.no_grad():
        # First pass: process the whole prompt at once
        out = model(input_ids, past_key_values=None, use_cache=True)
        logits = out.logits           # (1, prompt_len, vocab_size)
        past_key_values = out.past_key_values

        next_token = _sample(logits[0, -1, :], temperature)
        generated.append(next_token.item())

        # Subsequent passes: one token at a time, reusing KV cache
        for _ in range(n_tokens - 1):
            out = model(
                next_token.view(1, 1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits       # (1, 1, vocab_size)
            past_key_values = out.past_key_values
            next_token = _sample(logits[0, -1, :], temperature)
            generated.append(next_token.item())

    return generated


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()


def benchmark(model, tokenizer, prompt: str, n_tokens: int, device, temperature: float = 0.0):
    set_seed()
    # Warmup: avoids measuring one-time model initialization cost
    greedy_generate(model, tokenizer, prompt, 5, device, temperature)

    start = time.perf_counter()
    tokens = greedy_generate(model, tokenizer, prompt, n_tokens, device, temperature)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    tok_per_sec = n_tokens / elapsed
    return text, tok_per_sec, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The transformer architecture")
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    for model_name in ["distilgpt2", "gpt2-medium"]:
        print(f"Loading {model_name}...")
        model, tokenizer = load_model(model_name, device)

        text, tok_per_sec, elapsed = benchmark(
            model, tokenizer, args.prompt, args.n_tokens, device, args.temperature
        )

        print(f"  {tok_per_sec:.1f} tok/s  ({elapsed:.2f}s for {args.n_tokens} tokens)")
        print(f"  Output: {text[:100]}")
        print()
