"""
Generate figures for the README from benchmark data collected during this project.
All numbers are from runs on Apple M2 Max, recorded in progress.md and docs/optimizations.md.

Usage:  python tools/make_figures.py
Output: docs/figures/*.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def fig_optimization_journey():
    phases = [
        "Original\n(Day 8.5)",
        "fp16\n(Phase 1)",
        "torch.compile\n(Phase 2)",
        "Eager scheme\n(Phase 3)",
        "+ Sync cleanup\n(Phase 4)",
        "MLX greedy\n(Phase 5)*",
    ]
    speedups = [0.83, 0.80, 0.83, 1.08, 1.16, 2.69]
    kept     = [True,  False, False, True,  True,  True]
    colors   = ["#888888" if i == 0 else
                "#cc6666" if not kept[i] else
                "#5a9b5a" if i < 5 else "#4a90c9" for i in range(len(phases))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(phases, speedups, color=colors, edgecolor="black", linewidth=0.6)

    ax.axhline(1.0, ls="--", color="black", linewidth=0.8, alpha=0.5)
    ax.text(len(phases) - 0.5, 1.03, "greedy baseline = 1.0×",
            ha="right", va="bottom", fontsize=9, color="dimgray")

    for bar, val, k in zip(bars, speedups, kept):
        annotation = f"{val:.2f}×"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06,
                annotation, ha="center", fontsize=10, fontweight="bold")

    # legend-ish footnote
    fig.text(0.99, 0.01, "* MLX uses a different model pair (Qwen2.5 0.5B/1.5B 4-bit), shown for runtime comparison.",
             ha="right", fontsize=8, style="italic", color="dimgray")

    ax.set_ylabel("Speedup vs PyTorch greedy baseline")
    ax.set_title("Optimization journey on Apple M2 Max\n0.83× → 1.16× via algorithm; 2.69× via runtime swap",
                 fontsize=12, pad=12)
    ax.set_ylim(0, 3.2)

    plt.tight_layout()
    out = OUT / "optimization_journey.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_profile_breakdown():
    phases = ["draft proposal", "verifier score", "setup_next\n(removed in Phase 3)", "prefill (one-time)", "cache mgmt", "accept / reject"]
    pct = [25.1, 34.1, 29.3, 4.2, 4.1, 3.1]
    colors = ["#5a9b5a", "#4a90c9", "#cc6666", "#bbbbbb", "#888888", "#888888"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.barh(phases, pct, color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()

    for bar, val in zip(bars, pct):
        ax.text(val + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("% of total speculative time (K=4, original cached impl)")
    ax.set_title("Where time goes per speculative iteration\n88% in model forward passes; setup_next was the target of Phase 3",
                 fontsize=12, pad=12)
    ax.set_xlim(0, 42)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = OUT / "profile_breakdown.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_alpha_heatmap():
    alpha = np.array([
        [67.55, 60.48, 54.05, 53.90],
        [53.98, 46.22, 48.62, 50.39],
        [40.24, 38.16, 30.70, 36.91],
        [25.28, 19.34, 16.92, 20.55],
    ])
    K = [1, 2, 4, 8]
    T = [0.0, 0.5, 1.0, 1.5]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(alpha, cmap="RdYlGn", vmin=15, vmax=70, aspect="auto")

    ax.set_xticks(range(len(T)))
    ax.set_xticklabels([f"T = {t}" for t in T])
    ax.set_yticks(range(len(K)))
    ax.set_yticklabels([f"K = {k}" for k in K])

    for i in range(len(K)):
        for j in range(len(T)):
            color = "white" if alpha[i, j] < 35 else "black"
            ax.text(j, i, f"{alpha[i, j]:.1f}%", ha="center", va="center",
                    fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Acceptance rate α", labelpad=12)

    ax.set_title("Acceptance rate α(K, T)\ndistilgpt2 (draft) + gpt2-medium (verifier)",
                 fontsize=12, pad=12)
    ax.grid(False)

    plt.tight_layout()
    out = OUT / "alpha_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_pytorch_vs_mlx():
    methods = ["Greedy", "Spec K=1", "Spec K=2", "Spec K=4", "Spec K=8"]
    pytorch = [44.5, 33.4, 45.0, 46.9, 44.2]
    mlx     = [119.3, 74.5, 75.1, 61.4, 44.6]

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars1 = ax.bar(x - width / 2, pytorch, width,
                   label="PyTorch + MPS  (distilgpt2 / gpt2-medium)",
                   color="#cc6666", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width / 2, mlx, width,
                   label="MLX  (Qwen2.5 0.5B / 1.5B, 4-bit)",
                   color="#4a90c9", edgecolor="black", linewidth=0.6)

    for bars in (bars1, bars2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                    f"{v:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Same hardware, different runtime\nMLX greedy beats every PyTorch configuration by ~2.7×",
                 fontsize=12, pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 140)

    plt.tight_layout()
    out = OUT / "pytorch_vs_mlx.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_optimization_journey()
    fig_profile_breakdown()
    fig_alpha_heatmap()
    fig_pytorch_vs_mlx()
