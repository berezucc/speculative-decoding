import argparse
import time
import torch
from utils import get_device, set_seed
from baseline import greedy_generate
from draft import DraftModel
from verifier import VerifierModel
from speculative import speculative_generate


# A long base passage so we can truncate to any prompt length up to ~600 tokens
BASE_TEXT = (
    "The history of artificial intelligence research begins in the mid twentieth century, "
    "when a small group of mathematicians and engineers began to imagine machines that could think. "
    "Early efforts focused on symbolic reasoning, with programs that could prove theorems and play "
    "checkers. By the 1980s, attention had shifted toward statistical methods and neural networks. "
    "The introduction of the backpropagation algorithm allowed deeper networks to be trained, but "
    "computational limitations and data scarcity meant progress was slow. In the 2010s, the "
    "combination of large datasets, GPU acceleration, and architectural innovations such as the "
    "transformer transformed the field. Models began to demonstrate capabilities that had long been "
    "considered far off, including coherent text generation, image synthesis, and complex reasoning. "
    "Speculative decoding emerged as a technique to accelerate inference for these large autoregressive "
    "models, exploiting the observation that smaller models can often guess what larger models would "
    "produce, freeing the larger model to verify multiple guesses in parallel rather than generating "
    "tokens one at a time. The mathematical guarantee is striking: the output distribution is identical "
    "to sampling from the large model alone, but the wall clock time can be significantly reduced. "
    "Variants of the technique include tree based speculation, multi step rejection, and draft model "
    "fine tuning to maximize agreement with the verifier. Each variant offers different trade offs "
    "between implementation complexity and achievable speedup. Practitioners deploying these systems "
    "must balance theoretical predictions with hardware realities such as memory bandwidth, kernel "
    "launch overhead, and synchronization costs across heterogeneous compute units. Modern inference "
    "stacks like vLLM and TensorRT-LLM incorporate sophisticated optimizations such as paged attention, "
    "continuous batching, and operator fusion to extract maximum throughput from accelerators. These "
    "systems treat the prefill and decode phases of generation as fundamentally different workloads, "
    "with prefill being compute bound and decode being memory bandwidth bound. The asymmetry creates "
    "opportunities for techniques like chunked prefill and disaggregated serving, where the two phases "
    "run on different hardware tuned to their respective bottlenecks. Researchers continue to explore "
    "how speculative decoding combines with these systems, including draft models trained jointly with "
    "their verifiers, hierarchical speculation with multiple draft tiers, and learned routing that "
    "decides on the fly when speculation is worth attempting. The interplay between algorithmic "
    "guarantees and engineering realities defines much of modern machine learning systems work. "
    "Performance numbers in papers should always be read with attention to the hardware target, the "
    "software stack used, the batch size, the prompt length distribution, and the sampling temperature. "
    "Reproducing reported speedups outside the original setting requires careful matching of all these "
    "variables, and even then results often differ in ways that reveal hidden assumptions in the work. "
    "This makes empirical replication a high value activity, often more informative than yet another "
    "incremental method paper. Apple Silicon presents a distinctive target, with unified memory that "
    "removes some bottlenecks while introducing others, and a Metal Performance Shaders backend that "
    "is still maturing relative to CUDA. Tasks that look trivial on a discrete GPU sometimes require "
    "rethinking on Apple hardware, and vice versa. "
)


def make_prompt(tokenizer, target_length: int) -> str:
    ids = tokenizer.encode(BASE_TEXT)
    if len(ids) < target_length:
        raise ValueError(f"base text only {len(ids)} tokens, need {target_length}")
    truncated_ids = ids[:target_length]
    return tokenizer.decode(truncated_ids)


def time_greedy(verifier, prompt, n_tokens, device):
    set_seed()
    greedy_generate(verifier.model, verifier.tokenizer, prompt, 5, device, 0.0)
    start = time.perf_counter()
    greedy_generate(verifier.model, verifier.tokenizer, prompt, n_tokens, device, 0.0)
    return n_tokens / (time.perf_counter() - start)


def time_speculative(draft, verifier, prompt, n_tokens, k, device):
    set_seed()
    speculative_generate(draft, verifier, prompt, 5, k=k, temperature=0.0, device=device)
    start = time.perf_counter()
    _, stats = speculative_generate(draft, verifier, prompt, n_tokens, k=k, temperature=0.0, device=device)
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed, stats["acceptance_rate"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}\n")

    print("loading models...")
    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)
    print()

    rows = []  # list of (length, baseline_tps, {k: (tps, alpha)})

    for length in args.lengths:
        prompt = make_prompt(verifier.tokenizer, length)
        print(f"prompt length {length}")

        baseline_tps = time_greedy(verifier, prompt, args.n_tokens, device)
        print(f"  greedy: {baseline_tps:.1f} tok/s")

        per_k = {}
        for k in args.k_values:
            tps, alpha = time_speculative(draft, verifier, prompt, args.n_tokens, k, device)
            speedup = tps / baseline_tps
            per_k[k] = (tps, alpha, speedup)
            print(f"  K={k}: {tps:.1f} tok/s, alpha={alpha:.2%}, speedup={speedup:.2f}x")

        rows.append((length, baseline_tps, per_k))
        print()

    # final table
    print("=" * 72)
    print(f"Speedup vs greedy baseline by prompt length (temp=0, gen={args.n_tokens})")
    print("=" * 72)
    header = f"{'length':>8} | {'baseline':>10} | " + " | ".join(f"{'K='+str(k):>14}" for k in args.k_values)
    print(header)
    print("-" * len(header))
    for length, baseline_tps, per_k in rows:
        row = f"{length:>8} | {baseline_tps:>7.1f} tps | "
        cells = []
        for k in args.k_values:
            tps, alpha, speedup = per_k[k]
            cells.append(f"{speedup:.2f}x ({alpha:.0%})")
        row += " | ".join(f"{c:>14}" for c in cells)
        print(row)
