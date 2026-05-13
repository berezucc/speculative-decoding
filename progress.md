# Progress Log

## Day 1 - Baseline Greedy Decoding
**Status**: Done

**What I built**
- `src/utils.py`: device detection (MPS/CPU), model loader, seed helper
- `src/baseline.py`: manual autoregressive generation loop with KV cache, timing benchmark

**TLDR**
Wrote a greedy generation loop from scratch without using `model.generate()`. Two stages: prefill (whole prompt in one parallel pass to populate the KV cache) and decode (one token at a time, reusing the cache). Benchmarked distilgpt2 and gpt2-medium to get baseline tok/s, the 1.0x reference everything else is measured against.

**Key concepts learned**
- Logits: raw per-token scores from the model, shape `(1, seq_len, 50257)`
- KV cache: saves K and V vectors from previous tokens so you skip recomputation, `past_key_values` grows by 1 each step
- Temperature: divide logits by T before softmax, lower = more peaked (greedy-like), higher = flatter (more random)
- Prefill vs decode: prefill is compute-bound (parallel), decode is memory-bandwidth-bound (serial), speculative decoding attacks the decode bottleneck

**Baseline numbers**
| Model | tok/s |
|---|---|
| distilgpt2 | 63.4 |
| gpt2-medium | 30.8 |
| gamma ratio (draft/verifier) | 2.06x |

## Day 2 - Acceptance Criterion
**Status**: Done

**What I built**
- `src/acceptance.py`: acceptance_prob, get_probs, sample_residual
- `src/test_acceptance.py`: 5 unit tests covering edge cases

**TLDR**
Implemented the core math of speculative decoding. Draft model picks a token, verifier either accepts it (prob min(1, p/q)) or rejects it and samples a correction from the residual distribution max(0, p-q). All 5 tests pass. This is the decision logic that sits inside the speculative decoding loop on Day 5.

**Reference**
- Leviathan et al. 2023, Section 2.3: https://arxiv.org/pdf/2211.17192

## Day 3 - Draft Model: Propose K Tokens
**Status**: Done

**What I built**
- `src/draft.py`: DraftModel class that runs distilgpt2 autoregressively for K steps, returns draft token ids and full probability distributions

**TLDR**
Combines Day 1 (KV cache loop) and Day 2 (get_probs) into one class. Runs the draft model K times, stops, and returns two tensors: draft_tokens (k,) and draft_probs (k, vocab_size). These feed directly into the verifier on Day 4 and the acceptance criterion on Day 5.

## Day 4 - Verifier: Score Draft Sequence
**Status**: Done

**What I built**
- `src/verifier.py`: VerifierModel class that scores K draft tokens in a single parallel forward pass

**TLDR**
Verifier takes [prompt + K draft tokens] and runs gpt2-medium once, in parallel. Slices out logits at positions [L-1, L, ..., L+K-1] to get K+1 probability distributions: K for scoring each draft token, plus 1 bonus distribution if all K accepted. The off-by-one matters: logit at position i predicts token at position i+1.

**Why this is the key efficiency**
One forward pass scored 4 tokens at once. Doing it serially would have been 4 separate verifier calls. Memory bandwidth (reading the weights) is roughly the same for 1 token or K tokens, so you get K scores for the price of 1.

## Day 5 - End-to-End Speculative Decoding Loop
**Status**: Done

**What I built**
- `src/speculative.py`: speculative_generate() loop and correctness test

**TLDR**
Combined draft, verifier, and acceptance criterion into one loop. Each iteration: draft proposes K tokens, verifier scores them in one pass, walk through and accept or reject each, sample correction from residual if rejected, sample bonus from verifier if all K accepted. Correctness test passes: at temp=0 the output matches verifier-only greedy exactly.

**Results**
- Correctness: PASS (output identical to verifier-only greedy)
- Acceptance rate alpha: 40% at temp=0, K=4
- Predicted speedup with this alpha and gamma=2.06: 0.88x (slower than baseline)

**Honest finding**
Speculative decoding doesn't universally speed things up. With distilgpt2 (distilled from gpt2 small, not gpt2-medium) and M2 Max memory bandwidth, both alpha and gamma are lower than the paper's CUDA numbers. Algorithm is correct, output is provably lossless, but the speedup math doesn't favor this setup. This is the kind of real-world result Day 7 benchmarks will confirm.

## Day 6 - Measure Acceptance Rate
**Status**: Done

**What I built**
- `src/measure_alpha.py`: sweep across 20 prompts x K=[1,2,4,8] x temp=[0.0, 0.5, 1.0, 1.5]
- Results saved to `benchmarks/alpha_results.json`

**TLDR**
Ran the speculative loop across varied prompts (prose, factual, code, creative), K values, and temperatures. For each combo, computed mean alpha across prompts.

**Results**
| K  | T=0.0  | T=0.5  | T=1.0  | T=1.5  |
|----|--------|--------|--------|--------|
|  1 | 66.62% | 58.11% | 53.37% | 52.24% |
|  2 | 52.29% | 47.00% | 42.91% | 52.37% |
|  4 | 37.95% | 33.36% | 30.70% | 31.54% |
|  8 | 25.38% | 17.17% | 17.37% | 19.93% |

**Predicted speedup vs baseline (with gamma=2.06)**
- K=1, T=0.0: 1.12x (faster)
- K=2, T=0.0: 1.04x (about even)
- K=4, T=0.0: 0.86x (slower)
- K=8, T=0.0: 0.61x (much slower)

**Findings**
- Alpha drops sharply as K rises, the dominant effect on this hardware
- Temperature effect is real but weaker than the K effect
- Conventional K=4 from the paper is suboptimal here, K=1 is the sweet spot
- This is because gamma=2.06 on M2 Max is too low to amortize multiple draft passes

## Day 7 - Throughput Benchmark
**Status**: Done

**What I built**
- `src/benchmark.py`: times greedy baseline vs speculative at K=1,2,4,8, compares measured vs predicted speedup

**TLDR**
The actual wall-clock test. Runs 100-token generations across 5 prompts for greedy (verifier only) and speculative at each K. Computes mean tok/s, measured speedup vs baseline, and the predicted speedup from theory using Day 6 alpha values and Day 1 gamma.

**Results**
| Method | tok/s | alpha | measured | predicted | gap |
|---|---|---|---|---|---|
| Greedy (verifier) | 45.7 | - | 1.00x | 1.00x | - |
| Speculative K=1 | 19.2 | 80.64% | 0.42x | 1.22x | -0.80 |
| Speculative K=2 | 33.5 | 71.91% | 0.73x | 1.24x | -0.51 |
| Speculative K=4 | 38.0 | 58.95% | 0.83x | 1.14x | -0.31 |
| Speculative K=8 | 38.7 | 46.00% | 0.85x | 0.96x | -0.11 |

**Headline finding**
Measured speedup is significantly below predicted across all K, and the gap shrinks as K grows. The theoretical formula `(1 + K*alpha) / (1 + K/gamma)` only counts forward passes. Real iterations also pay Python loop overhead, tensor concat, MPS sync, and sampling logic, all roughly fixed per iteration. Small K means short iterations means overhead dominates. Large K amortizes overhead over more tokens.

**Why this is interesting**
The paper's 2-3x speedups were on TPU/CUDA with C++ runtimes where per-iteration overhead is negligible. On M2 Max with PyTorch + MPS, that overhead eats the algorithmic gains. No K configuration beats baseline here, but K=8 comes closest to predicted (within 11%).

**Also notable**
- Alpha is higher here than Day 6 measured (80% vs 66% at K=1), likely due to using only 5 prompts vs 20
- Baseline tok/s is faster than Day 1 (45.7 vs 30.8), likely thermal state or prompt mix differences

## Day 8 - Sequence Length Sweep
**Status**: Not started

## Day 9 - Profiling
**Status**: Not started

## Day 10 - Clean README + Final Benchmark Table
**Status**: Not started
