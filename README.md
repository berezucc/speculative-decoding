# Speculative Decoding

From-scratch implementation of speculative decoding (Leviathan et al. 2023) in PyTorch, benchmarked on Apple M2 Max.

**Status**: Complete

---

## What is speculative decoding?

Autoregressive LLM inference is serial: each token requires one full forward pass of the large model. For a 355M parameter model, that means reading ~700MB of weights per token, which is memory-bandwidth bound, not compute bound.

Speculative decoding breaks the bottleneck:

1. A small **draft model** proposes K tokens cheaply (fast, lower quality)
2. The large **verifier model** scores all K tokens in a single parallel forward pass
3. Tokens are accepted or rejected based on the probability ratio `p(x) / q(x)`
4. The output distribution is **identical** to pure verifier sampling (lossless)

```
prompt -> draft proposes [t1, t2, t3, t4]
       -> verifier scores all 4 in one pass
       -> accept t1, t2; reject t3 -> sample correction c
       -> output: t1, t2, c (3 new tokens for 1 verifier pass)
```

---

## Model pair

| Role | Model | Parameters |
|---|---|---|
| Draft | distilgpt2 | 82M |
| Verifier | gpt2-medium | 355M |

Both use the same GPT-2 BPE tokenizer (50257 vocab), required for the acceptance criterion.

---

## Headline results (M2 Max, MPS backend, PyTorch)

**Correctness**: PASS. Output at temperature=0 matches verifier-only greedy decoding exactly.

**Throughput** (100 tokens, 5 varied prompts, temp=0, after eager-scheme optimization):

| Method | tok/s | Speedup |
|---|---|---|
| Greedy (verifier only) | 41.1 | 1.00x |
| Speculative K=4 | 44.5 | **1.08x** |
| Speculative K=8 | 44.2 | 1.07x |

The eager scheme removed the end-of-iter "setup_next" forward passes (29% of total time in the original cached impl per Day 9 profile), buying 1 fewer forward pass per speculative iteration. That single change crossed the 1.0x threshold on M2 Max + PyTorch + MPS.

**Acceptance rate by K and temperature**:

| K  | T=0.0  | T=0.5  | T=1.0  | T=1.5  |
|----|--------|--------|--------|--------|
|  1 | 67.55% | 60.48% | 54.05% | 53.90% |
|  2 | 53.98% | 46.22% | 48.62% | 50.39% |
|  4 | 40.24% | 38.16% | 30.70% | 36.91% |
|  8 | 25.28% | 19.34% | 16.92% | 20.55% |

**Sequence length effect** (K=4, temp=0):

| Prompt length | tok/s | Speedup |
|---|---|---|
| 32 | 42.0 | 0.91x |
| 128 | 28.3 | 0.66x |
| 512 | 33.3 | 0.86x |

---

## MLX comparison

| Runtime | Model pair | Greedy tok/s | Best speculative tok/s | Speedup |
|---|---|---|---|---|
| PyTorch + MPS | distilgpt2 / gpt2-medium | 44.5 | 46.9 (K=4) | 1.08x |
| MLX | Qwen2.5 0.5B / 1.5B (4-bit) | 119.3 | 75.1 (K=2) | 0.63x |

MLX greedy is **2.7x faster than PyTorch greedy** on the same hardware. Speculative decoding inside MLX is actually slower than MLX greedy for this 4-bit quantized pair (gamma too small to amortize the algorithm's overhead). The runtime is the bigger lever than the algorithm on M2 Max.

## Three findings

**What surprised me**: the original cached implementation never beat greedy baseline on M2 Max + PyTorch + MPS, despite correct algorithm and proper KV cache management. The theoretical 1.2x speedup at K=4 vanished against per-iteration overhead. The fix wasn't algorithmic; it was reorganizing the loop to do 1 fewer forward pass per iteration.

**Where the bottleneck was**: not the verifier forward pass, but the fixed per-call cost of MPS kernel launches. Profiling shows ~25 ms fixed overhead per verifier forward pass and ~9 ms per draft pass, regardless of input length. The original cached scheme did 6 forward passes per iter at K=4 (1 saved-logit prep + 3 draft proposal + 1 verifier score + 2 setup-next) for ~3 tokens. Eager scheme cuts that to 5 by feeding the prior-iter's last token at the start of each iter instead of saving its logit at the end.

**When speculative decoding helps**: it now does, on M2 Max. The eager scheme reaches 1.08x at K=4 and 1.07x at K=8. On hardware with lower per-call overhead (CUDA + C++ runtime, MLX), the speedup would be larger because fixed cost per pass approaches zero. On PyTorch + MPS specifically, the win is small and requires careful elimination of per-iteration overhead. Float16 inference and `torch.compile` both failed to help here: fp16 doesn't matter when overhead is fixed-per-call, and torch.compile on MPS errors on LayerNorm lowering for these models.

---

## A bug worth flagging

The first speculative implementation reprocessed the entire prompt on every iteration (no KV cache persistence). At length 512, K=8 measured 0.28x of baseline. Fixing the KV cache management lifted that to 0.74x without changing any algorithm. The correctness test (output matches verifier-only greedy at temp=0) caught nothing because the bug was an efficiency issue, not a correctness one. Lesson: profile and benchmark do not converge on the same kind of bug.

---

## How to run

```bash
pip install -r requirements.txt

# Baseline (greedy, no speculation)
python src/baseline.py --prompt "The transformer architecture" --n_tokens 100

# Speculative decoding (cached, fast)
python src/speculative_cached.py --prompt "The transformer architecture" --n_tokens 100 --k 4

# Correctness test (must pass before benchmarking)
python src/speculative_cached.py --test

# Throughput benchmark
python src/benchmark.py --n_tokens 100

# Acceptance rate sweep
python src/measure_alpha.py --n_tokens 30 --n_prompts 20

# Sequence length sweep
python src/seq_length_sweep.py --n_tokens 100

# Profiling breakdown
python src/profile_breakdown.py
```

---

## Project structure

```
src/
  baseline.py            greedy decoding with manual KV cache loop
  draft.py               draft model: propose K tokens
  verifier.py            verifier model: score K tokens in one pass
  acceptance.py          acceptance criterion + residual sampling
  speculative.py         first speculative loop (no cache persistence)
  speculative_cached.py  cached speculative loop (correct + efficient)
  benchmark.py           throughput vs greedy
  measure_alpha.py       acceptance rate sweep across K and temperature
  seq_length_sweep.py    speedup vs prompt length
  profile_breakdown.py   per-phase timing breakdown
  utils.py               model loader, device selection, seeding
docs/
  spec.md                project spec, sprint plan, reading list
benchmarks/
  alpha_results.json     raw acceptance rate data
progress.md              day-by-day log
```

---

## Papers

- Leviathan et al. 2023, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Chen et al. 2023, [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
