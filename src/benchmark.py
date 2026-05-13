import argparse
import time
import torch
from utils import get_device, set_seed
from baseline import greedy_generate
from draft import DraftModel
from verifier import VerifierModel
from speculative_cached import speculative_generate_cached as speculative_generate


PROMPTS = [
    "The transformer architecture",
    "Photosynthesis is the process by which",
    "def fibonacci(n):",
    "Once upon a time in a small village",
    "Climate change is one of the most",
]


def time_greedy(verifier, prompt, n_tokens, device):
    set_seed()
    # warmup
    greedy_generate(verifier.model, verifier.tokenizer, prompt, 5, device, 0.0)

    start = time.perf_counter()
    tokens = greedy_generate(verifier.model, verifier.tokenizer, prompt, n_tokens, device, 0.0)
    elapsed = time.perf_counter() - start
    return elapsed, n_tokens / elapsed


def time_speculative(draft, verifier, prompt, n_tokens, k, device):
    set_seed()
    # warmup
    speculative_generate(draft, verifier, prompt, 5, k=k, temperature=0.0, device=device)

    start = time.perf_counter()
    tokens, stats = speculative_generate(
        draft, verifier, prompt, n_tokens, k=k, temperature=0.0, device=device
    )
    elapsed = time.perf_counter() - start
    return elapsed, n_tokens / elapsed, stats["acceptance_rate"]


def predicted_speedup(alpha, k, gamma):
    return (1 + k * alpha) / (1 + k / gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=2.06, help="draft/verifier speed ratio from day 1")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}\n")

    print("loading models...")
    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)
    print()

    # baseline
    print("timing greedy baseline (verifier only)...")
    baseline_times = []
    for prompt in PROMPTS:
        elapsed, tps = time_greedy(verifier, prompt, args.n_tokens, device)
        baseline_times.append(tps)
        print(f"  {tps:6.1f} tok/s  ({elapsed:.2f}s)  {prompt[:40]!r}")
    baseline_tps = sum(baseline_times) / len(baseline_times)
    print(f"  mean: {baseline_tps:.1f} tok/s\n")

    # speculative at each K
    k_values = [1, 2, 4, 8]
    results = {}

    for k in k_values:
        print(f"timing speculative K={k}...")
        tps_list = []
        alpha_list = []
        for prompt in PROMPTS:
            elapsed, tps, alpha = time_speculative(draft, verifier, prompt, args.n_tokens, k, device)
            tps_list.append(tps)
            alpha_list.append(alpha)
            print(f"  {tps:6.1f} tok/s  ({elapsed:.2f}s)  alpha={alpha:.2%}  {prompt[:30]!r}")
        mean_tps = sum(tps_list) / len(tps_list)
        mean_alpha = sum(alpha_list) / len(alpha_list)
        results[k] = (mean_tps, mean_alpha)
        print(f"  mean: {mean_tps:.1f} tok/s, alpha {mean_alpha:.2%}\n")

    # summary table
    print("=" * 70)
    print("Throughput Summary (temp=0.0, n_tokens={})".format(args.n_tokens))
    print("=" * 70)
    print(f"{'Method':<20} {'tok/s':>10} {'alpha':>10} {'measured':>12} {'predicted':>12}")
    print(f"{'-' * 20} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12}")
    print(f"{'Greedy (verifier)':<20} {baseline_tps:>10.1f} {'-':>10} {'1.00x':>12} {'1.00x':>12}")
    for k in k_values:
        tps, alpha = results[k]
        measured = tps / baseline_tps
        predicted = predicted_speedup(alpha, k, args.gamma)
        print(f"{'Speculative K=' + str(k):<20} {tps:>10.1f} {alpha:>9.2%} {measured:>11.2f}x {predicted:>11.2f}x")
