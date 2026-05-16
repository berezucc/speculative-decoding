# Speculative Decoding on Apple M2 Max

A from-scratch PyTorch implementation of speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)), benchmarked and optimized for Apple Silicon. The implementation is provably lossless (output at temperature=0 matches verifier-only greedy decoding exactly) and includes a full optimization sequence taking the speedup from 0.83x to 1.16x of greedy baseline.

A second benchmark using Apple's MLX runtime reveals that the inference framework, not the algorithm, is the dominant factor on M2 Max: MLX greedy alone achieves 2.7x over PyTorch greedy.

## Headline results

All numbers from 100-token generation, 5 varied prompts, temperature=0, on Apple M2 Max.

| Configuration | Throughput | Speedup vs PyTorch greedy |
|---|---|---|
| PyTorch greedy (verifier only) | 44.5 tok/s | 1.00x |
| PyTorch speculative (K=4, optimized) | 46.9 tok/s | 1.06x |
| MLX greedy (Qwen2.5-1.5B, 4-bit) | 119.3 tok/s | **2.68x** |
| MLX speculative (Qwen2.5 0.5B/1.5B, 4-bit) | 75.1 tok/s | 1.69x |

**Correctness**: output at temperature=0 is bit-identical to verifier-only greedy decoding (mathematical guarantee of speculative sampling).

## TL;DR

- Implemented the algorithm correctly. KV cache management, the rejection / residual sampling step, and the off-by-one verifier scoring all work as the paper specifies.
- The naive cached implementation runs at 0.83x of greedy on PyTorch + MPS. A 5-phase optimization pass identified per-call kernel-launch overhead (~25 ms on MPS) as the bottleneck and lifted the speedup to 1.16x by reorganizing the iteration loop ("eager scheme") to do one fewer forward pass per iter.
- The bigger lever turns out to be the runtime. MLX greedy is 2.7x of PyTorch greedy on the same hardware. Inside MLX, speculative decoding does not help: 4-bit quantized models are already memory-bandwidth-fast, the draft/verifier cost ratio collapses, and the algorithm's overhead exceeds its benefit.

For the engineering detail behind these numbers, see [`docs/optimizations.md`](docs/optimizations.md).

## Model pair (PyTorch implementation)

| Role | Model | Parameters | Tokenizer |
|---|---|---|---|
| Draft | distilgpt2 | 82M | GPT-2 BPE (50257) |
| Verifier | gpt2-medium | 355M | GPT-2 BPE (50257) |

Shared tokenizer is required by the acceptance criterion (both `p(x)` and `q(x)` must refer to the same vocabulary).

## Quick start

```bash
pip install -r requirements.txt

# Correctness check (must pass before trusting any benchmark)
python src/speculative_cached.py --test

# Greedy baseline
python src/baseline.py --prompt "The transformer architecture" --n_tokens 100

# Speculative decoding
python src/speculative_cached.py --prompt "The transformer architecture" --n_tokens 100 --k 4

# Throughput benchmark
python src/benchmark.py --n_tokens 100

# MLX comparison (requires `pip install mlx mlx-lm`)
python src/benchmark_mlx.py --n_tokens 100
```

## Findings

**On the algorithm.** Speculative decoding is provably lossless and the implementation here demonstrates that on real hardware: at temperature=0 the speculative output matches verifier-only greedy decoding token-for-token. The acceptance rate α (fraction of draft tokens accepted) scales roughly as the paper predicts: 67% at K=1, dropping to 25% at K=8 on this model pair, and decreasing with temperature.

**On the bottleneck.** Profiling with `torch.mps.synchronize()` reveals ~25 ms of fixed cost per verifier forward pass on PyTorch + MPS regardless of input size, with marginal cost per token of only ~0.7-1 ms. This means the algorithm's per-iteration overhead (multiple forward passes for draft proposal, verifier scoring, and cache management) accumulates faster than the K-token parallelism saves. 88% of speculative iteration time is spent inside model forward passes, not in cache management or accept/reject logic.

**On optimization.** Within PyTorch+MPS, the biggest single win was reorganizing the loop to remove the end-of-iter "setup_next" forward passes (29% of total time per profile). fp16 inference did not help (bottleneck is not memory bandwidth on this runtime). torch.compile fails on the MPS backend for these models (`aten.var_mean.correction` has no lowering for LayerNorm).

**On runtime.** Switching from PyTorch+MPS to MLX (with the necessarily-different Qwen2.5 model pair) gives 2.7x speedup with no algorithmic change. Inside MLX, however, the built-in speculative decoding is slower than greedy: the draft/verifier cost ratio γ collapses when both models are small and 4-bit quantized, so there's nothing to amortize. **Speculative decoding helps when the verifier is slow.**

## Reproducing the results

All benchmarks are deterministic at temperature=0 with seed=42 (set in `src/utils.py:set_seed`).

```bash
# Correctness, throughput, and profile breakdown — PyTorch + MPS
python src/speculative_cached.py --test
python src/benchmark.py --n_tokens 100
python src/profile_breakdown.py

# Acceptance rate sweep across K and temperature
python src/measure_alpha.py --n_tokens 30 --n_prompts 20

# Sequence length sweep
python src/seq_length_sweep.py --n_tokens 100

# MLX baseline (requires pip install mlx mlx-lm)
python src/benchmark_mlx.py --n_tokens 100
```

Per-run variance on MPS is meaningful (±5-10% on absolute tok/s due to thermal state and process scheduling). The within-run speedup ratio is more stable than absolute numbers.

## Project structure

```
src/
  baseline.py            greedy decoding loop with manual KV cache
  draft.py               draft model wrapper: propose K tokens
  verifier.py            verifier model wrapper: score K tokens in one pass
  acceptance.py          acceptance criterion + residual sampling, unit-tested
  speculative.py         first speculative loop (no cache persistence) - kept for history
  speculative_cached.py  cached speculative loop with eager scheme (current best)
  benchmark.py           throughput benchmark, measured vs predicted speedup
  benchmark_mlx.py       MLX comparison benchmark
  measure_alpha.py       acceptance rate sweep across K and temperature
  seq_length_sweep.py    speedup vs prompt length
  profile_breakdown.py   per-phase timing breakdown of a speculative iteration
  test_acceptance.py     unit tests for the acceptance criterion
  utils.py               model loading, device selection, seed control

docs/
  spec.md                project spec, sprint plan, reading list
  optimization-plan.md   5-phase optimization plan with stop conditions
  optimizations.md       optimization deep-dive: hypotheses, code diffs, verdicts

benchmarks/
  alpha_results.json     raw acceptance rate data (K x temperature sweep)
  results.md             benchmark tables

progress.md              day-by-day log of the 10-day sprint + optimization pass
```

## References

- Leviathan, Y., Kalman, M., Matias, Y. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML.
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., Jumper, J. (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). arXiv:2302.01318.

## License

MIT.
