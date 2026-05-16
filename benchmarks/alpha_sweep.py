import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import get_device, set_seed
from draft import DraftModel
from verifier import VerifierModel
from speculative import speculative_generate


PROMPTS = [
    # prose
    "The transformer architecture",
    "She opened the door and saw",
    "Once upon a time in a small village",
    "The detective examined the evidence carefully and",
    "Climate change is one of the most",

    # factual
    "The capital of France is",
    "Photosynthesis is the process by which",
    "World War II ended in",
    "The speed of light in vacuum is approximately",
    "Albert Einstein developed the theory of",

    # code-ish
    "def fibonacci(n):",
    "import numpy as np\n",
    "function isPrime(n) {",
    "SELECT * FROM users WHERE",
    "# Python script to read a CSV file\n",

    # creative
    "The dragon soared above the",
    "In the year 3024, humanity had",
    "Music has always been a way to",
    "If I could travel anywhere, I would",
    "The most important lesson I learned",
]


def run_sweep(draft, verifier, device, prompts, k_values, temperatures, n_tokens):
    results = {}  # (k, temp) -> list of alphas

    total_runs = len(prompts) * len(k_values) * len(temperatures)
    run_idx = 0

    for temp in temperatures:
        for k in k_values:
            alphas = []
            for prompt in prompts:
                run_idx += 1
                set_seed(42)
                _, stats = speculative_generate(
                    draft, verifier, prompt, n_tokens=n_tokens,
                    k=k, temperature=temp, device=device,
                )
                alphas.append(stats["acceptance_rate"])
                print(f"  [{run_idx}/{total_runs}] k={k} temp={temp} prompt={prompt[:40]!r:40s} alpha={stats['acceptance_rate']:.2%}")
            results[(k, temp)] = alphas

    return results


def print_table(results, k_values, temperatures):
    print("\n" + "=" * 60)
    print("Mean acceptance rate by K and Temperature")
    print("=" * 60)

    header = "| K  | " + " | ".join(f"T={t}" for t in temperatures) + " |"
    sep = "|----|" + "|".join("--------" for _ in temperatures) + "|"
    print(header)
    print(sep)

    for k in k_values:
        row = f"| {k:2d} | "
        for t in temperatures:
            mean_alpha = sum(results[(k, t)]) / len(results[(k, t)])
            row += f"{mean_alpha:6.2%} | "
        print(row)


def save_results(results, path):
    serializable = {f"k={k},temp={t}": alphas for (k, t), alphas in results.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nraw results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tokens", type=int, default=30, help="tokens per run")
    parser.add_argument("--n_prompts", type=int, default=10, help="number of prompts to use")
    parser.add_argument("--output", type=str, default="benchmarks/alpha_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}\n")

    print("loading models...")
    draft = DraftModel("distilgpt2", device)
    verifier = VerifierModel("gpt2-medium", device)

    prompts = PROMPTS[: args.n_prompts]
    k_values = [1, 2, 4, 8]
    temperatures = [0.0, 0.5, 1.0, 1.5]

    print(f"\nsweeping {len(prompts)} prompts x {len(k_values)} K x {len(temperatures)} temps "
          f"= {len(prompts) * len(k_values) * len(temperatures)} runs\n")

    results = run_sweep(draft, verifier, device, prompts, k_values, temperatures, args.n_tokens)
    print_table(results, k_values, temperatures)
    save_results(results, args.output)
