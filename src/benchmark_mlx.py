"""
MLX speculative decoding benchmark, for comparison against the PyTorch+MPS implementation.

Model pair: Qwen2.5-0.5B-Instruct-4bit (draft) and Qwen2.5-1.5B-Instruct-4bit (verifier).
Not the same models as the PyTorch benchmark (gpt2 isn't pre-converted to MLX), but a similar
size ratio. The headline number to compare is the speedup ratio (spec / greedy) within each
framework, not the absolute tok/s across frameworks.
"""
import argparse
import time
from mlx_lm import load, stream_generate


PROMPTS = [
    "The transformer architecture",
    "Photosynthesis is the process by which",
    "def fibonacci(n):",
    "Once upon a time in a small village",
    "Climate change is one of the most",
]


def time_run(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft_tokens=4):
    kwargs = {}
    if draft_model is not None:
        kwargs["num_draft_tokens"] = num_draft_tokens

    # warmup
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=5,
                             draft_model=draft_model, **kwargs):
        pass

    start = time.perf_counter()
    n_tokens = 0
    n_from_draft = 0
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                draft_model=draft_model, **kwargs):
        n_tokens += 1
        if draft_model is not None and getattr(resp, "from_draft", False):
            n_from_draft += 1
    elapsed = time.perf_counter() - start

    return {
        "tok_per_sec": n_tokens / elapsed,
        "elapsed": elapsed,
        "n_tokens": n_tokens,
        "n_from_draft": n_from_draft,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--verifier", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--draft", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    args = parser.parse_args()

    print(f"loading verifier ({args.verifier})...")
    verifier, tok = load(args.verifier)
    print(f"loading draft ({args.draft})...")
    draft, _ = load(args.draft)
    print()

    # Greedy baseline
    print("timing greedy (verifier only)...")
    base_tps = []
    for p in PROMPTS:
        r = time_run(verifier, tok, p, args.n_tokens)
        base_tps.append(r["tok_per_sec"])
        print(f"  {r['tok_per_sec']:6.1f} tok/s  {p[:40]!r}")
    mean_base = sum(base_tps) / len(base_tps)
    print(f"  mean: {mean_base:.1f} tok/s\n")

    # Speculative at different K
    summary = {}
    for k in [1, 2, 4, 8]:
        print(f"timing speculative K={k}...")
        spec_tps, alphas = [], []
        for p in PROMPTS:
            r = time_run(verifier, tok, p, args.n_tokens, draft_model=draft, num_draft_tokens=k)
            spec_tps.append(r["tok_per_sec"])
            alpha = r["n_from_draft"] / r["n_tokens"] if r["n_tokens"] else 0
            alphas.append(alpha)
            print(f"  {r['tok_per_sec']:6.1f} tok/s  alpha={alpha:.2%}  {p[:40]!r}")
        mts = sum(spec_tps) / len(spec_tps)
        ma = sum(alphas) / len(alphas)
        summary[k] = (mts, ma)
        print(f"  mean: {mts:.1f} tok/s, alpha {ma:.2%}\n")

    print("=" * 60)
    print(f"MLX Speculative Decoding (Qwen2.5 pair, n_tokens={args.n_tokens})")
    print("=" * 60)
    print(f"{'Method':<20} {'tok/s':>10} {'alpha':>10} {'speedup':>10}")
    print(f"{'Greedy':<20} {mean_base:>10.1f} {'-':>10} {'1.00x':>10}")
    for k, (tps, alpha) in summary.items():
        speedup = tps / mean_base
        print(f"{'Speculative K=' + str(k):<20} {tps:>10.1f} {alpha:>9.2%} {speedup:>9.2f}x")
