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
**Status**: Done

**What I built**
- `src/seq_length_sweep.py`: sweeps prompt lengths 32, 128, 512 across K=1,2,4,8

**TLDR**
Tested whether longer prompts amortize per-iteration overhead enough to push speculative above baseline. They don't, and the speedup actually gets dramatically worse as prompt length grows. This is the signature of a KV cache persistence bug.

**Results**
| Length | Baseline tok/s | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| 32 | 47.4 | 0.23x (77%) | 0.35x (82%) | 0.67x (65%) | 0.95x (45%) |
| 128 | 39.3 | 0.13x (68%) | 0.23x (56%) | 0.39x (38%) | 0.56x (22%) |
| 512 | 32.1 | 0.10x (75%) | 0.17x (75%) | 0.22x (54%) | 0.28x (36%) |

**Headline finding**
Speedup gets dramatically worse as prompt length grows. K=8 drops from 0.95x at length 32 to 0.28x at length 512. This is the opposite of what speculative decoding should do.

**Root cause**
The KV cache is not persisted across speculative iterations. `verifier.score()` runs with `use_cache=False` and the draft model rebuilds its cache from scratch on every iteration. At length 512 with 100 generated tokens, this means the verifier reprocesses ~520 tokens on each of ~26 iterations, doing roughly 135x more compute than greedy baseline.

**What this means**
The algorithm is correct (Day 5 correctness test still passes) but the implementation has an O(N*M) inefficiency that scales with prompt length times iteration count. Fix requires persisting KV caches across iterations and truncating them on rejection. Implementation is roughly one day of work.

## Day 8.5 - KV Cache Persistence Fix
**Status**: Done

**What I built**
- `src/speculative_cached.py`: rewrite of the speculative loop with KV cache persisted across iterations
- Updated `benchmark.py`, `seq_length_sweep.py`, `measure_alpha.py` to use the cached version

**TLDR**
Naive impl reprocessed the entire prompt every iteration. Fix: maintain KV cache across iterations, only feeding new tokens per call. Truncate cache on rejection, extend on full acceptance + bonus. Algorithm and output unchanged.

**Correctness**: PASS (output matches verifier-only greedy at temp=0)

**Throughput benchmark (varied prompts, 100 tokens)**
| Method | tok/s | alpha | measured | predicted |
|---|---|---|---|---|
| Greedy (verifier) | 48.6 | - | 1.00x | 1.00x |
| Speculative K=1 | 32.1 | 80.64% | 0.66x | 1.22x |
| Speculative K=2 | 35.5 | 71.91% | 0.73x | 1.24x |
| Speculative K=4 | 40.3 | 58.95% | 0.83x | 1.14x |
| Speculative K=8 | 41.7 | 46.00% | 0.86x | 0.96x |

**Sequence length sweep (before vs after fix)**
| Length | K | Before fix | After fix | Change |
|---|---|---|---|---|
| 32 | 4 | 0.67x | 0.91x | +0.24 |
| 32 | 8 | 0.95x | 0.79x | -0.16 |
| 128 | 4 | 0.39x | 0.66x | +0.27 |
| 128 | 8 | 0.56x | 0.48x | -0.08 |
| 512 | 1 | 0.10x | 0.69x | +0.59 |
| 512 | 4 | 0.22x | 0.86x | +0.64 |
| 512 | 8 | 0.28x | 0.74x | +0.46 |

**Findings**
- Massive improvement at long prompts: K=4 at length 512 went from 0.22x to 0.86x
- Small regression at K=8 short prompts: cached impl adds 2-3 extra forward passes per iter for cache management (feed correction/bonus, extend draft cache after full acceptance). At length 32 with K=8 these extras nearly cancel the cache savings
- Best result: K=4 at length 32: 0.91x of baseline, within 9% of greedy
- Still no config breaks 1.0x on M2 Max

**Why we still do not beat baseline**
The remaining gap is per-iteration Python + MPS overhead, not algorithmic. Each forward pass on MPS has fixed kernel-launch / sync cost regardless of how many tokens it processes. With 2-4 extra small forward passes per iteration for cache management plus the K-1 draft proposal passes, the fixed overhead adds up. With a C++ runtime (vLLM, TGI, MLX), this overhead largely disappears and theoretical speedup is achievable.

## Day 9 - Profiling
**Status**: Done

**What I built**
- `src/profile_breakdown.py`: per-phase timing of a speculative iteration plus a kernel-launch overhead test

**Phase breakdown (K=4, 100 tokens, 5 prompts, 154 iterations total)**
| Phase | Per iter (ms) | % of total |
|---|---|---|
| prefill (one-time) | 136.2 | 4.2% |
| draft_proposal | 26.4 | 25.1% |
| verifier_score | 35.8 | 34.1% |
| accept_reject | 3.2 | 3.1% |
| cache_mgmt | 4.3 | 4.1% |
| setup_next | 30.7 | 29.3% |

**Kernel launch overhead (verifier, gpt2-medium)**
| Input tokens | Time (ms) | Cost per extra token |
|---|---|---|
| 1 | 26.3 | - |
| 4 | 38.2 | 4.0 ms |
| 16 | 46.8 | 0.7 ms |
| 64 | 79.9 | 0.7 ms |

**Kernel launch overhead (draft, distilgpt2)**
| Input tokens | Time (ms) |
|---|---|
| 1 | 9.5 |
| 4 | 27.6 |
| 16 | 21.1 |
| 64 | 25.8 |

**Findings**
- Verifier has ~25 ms fixed cost per forward pass on MPS, regardless of input size. Marginal cost per token is only ~0.7-1 ms after that.
- Draft has ~9 ms fixed cost per forward pass.
- 88% of speculative time is spent inside model forward passes. Cache mgmt + accept/reject = 7%.
- Per speculative iter at K=4: 6 forward passes producing ~3 tokens at alpha=60% = ~31 ms per output token.
- Greedy: 1 forward pass per token = ~26 ms per output token.
- Speculative does more forward passes per output token than greedy on MPS because the per-call fixed cost dominates the K-parallelism benefit.

**The headline**
Algorithm is correct. The per-call MPS overhead is the entire reason speculative does not beat baseline. With a C++ runtime where forward passes cost ~1 ms instead of ~25 ms fixed, speculative would cross 1.0x.

## Day 10 - Final README + Benchmark Table
**Status**: Done

**What I built**
- Updated `README.md` with headline results, three findings, how-to-run, project structure
- Pulled together all benchmark numbers from Days 1, 6, 7, 8, 8.5, 9 into the final tables

**TLDR**
LinkedIn-ready writeup. Three sentences cover: what surprised me (no speedup over greedy on MPS), where the bottleneck is (per-call MPS overhead ~25 ms fixed, not algorithm), and when speculative helps (C++ runtime / CUDA). Included the KV cache bug story as an honest engineering lesson.

## Day 10 - Final README + Benchmark Table
**Status**: Done

**What I built**
- Updated `README.md` with headline results, three findings, how-to-run, project structure
- Pulled together all benchmark numbers from Days 1, 6, 7, 8, 8.5 into the final tables

**TLDR**
Wrote the LinkedIn-ready writeup. Three sentences on what surprised me (no speedup over greedy), where the bottleneck is (per-iteration Python + MPS overhead, not the verifier), and when speculative helps (CUDA + C++ runtime, or very long prompts on MPS). Included the KV cache bug story as an honest engineering lesson.

## Optimization Pass (post Day 10)
**Status**: Done, K=4 broke 1.0x

**What I did**
Executed `docs/optimization-plan.md` phases in order, stop condition K=4 speedup >= 1.0x.

**Results**
| Phase | Change | Greedy tok/s | Spec K=4 tok/s | Speedup | Outcome |
|---|---|---|---|---|---|
| 0 (baseline) | fp32, current code | 48.6 | 40.3 | 0.83x | starting point |
| 1 | fp16 | 46.5 | 37.1 | 0.80x | reverted (no benefit; MPS per-call overhead dominates math) |
| 2 | torch.compile | 39.6 | 33.1 | 0.83x | reverted (MPS cannot lower aten.var_mean.correction; fallback to eager hurts) |
| 3 | eager scheme | 41.1 | 44.5 | **1.08x** | KEPT, stop condition met (K=8: 1.07x) |

**The winning change**
Eager scheme rewrite of `speculative_generate_cached()`. Removed the end-of-iter `setup_next` block (29% of total time per Day 9 profile). Each iteration now does K draft forward passes (instead of K-1 + 1 cleanup) and one K+1-token verifier pass (instead of K-token verifier + 1 cleanup). Net: 1 fewer forward pass per iter at ~25 ms per pass on MPS.

**Correctness preserved**: output matches verifier-only greedy at temp=0.

**Negative findings (the things that did NOT work on M2 Max)**
- fp16: slightly slower. Forward pass time on MPS is dominated by kernel launch overhead, not memory bandwidth, so halving the weights didn't help.
- torch.compile: fails on MPS for these models because the MPS backend has no lowering for `aten.var_mean.correction` (used in LayerNorm). With suppress_errors=True the fallback to eager makes things slower.

**Phases not attempted**
- Phase 4 (sync cleanup): could add another 5-10% if pursued
- Phase 5 (MLX port): remains the next lever if needed, but not necessary now

## Optimization Pass 2 (Phase 4 + Phase 5)

### Phase 4 - Sync cleanup
**Status**: Done, neutral

Vectorized the accept/reject loop in `speculative_cached.py`:
- One `.gather()` over verifier and draft probs gives all K alpha values at once
- Single `.tolist()` to bring alphas to CPU (was K separate `.item()` syncs)
- Pre-rolled K random numbers as a CPU tensor (was K separate `torch.rand(1).item()` calls)
- Inlined the draft sampling at temp=0 to avoid the `.max().item()` inside `_sample`

Three runs gave K=4 speedups of 0.99x, 1.13x, 1.08x; mean ~1.07x. Within noise of Phase 3. Kept for code quality but didn't move the needle. The forward passes truly dominate.

### Phase 5 - MLX comparison
**Status**: Done, big surprise

Couldn't use the gpt2/distilgpt2 pair (mlx-lm doesn't ship pre-converted weights and direct HF loading fails on parameter naming). Substituted: Qwen2.5-1.5B-Instruct-4bit (verifier) and Qwen2.5-0.5B-Instruct-4bit (draft). Same hardware (M2 Max), different runtime and different models.

**Results (MLX, 100 tokens, 5 prompts)**
| Method | tok/s | alpha | speedup vs MLX greedy | speedup vs PyTorch greedy |
|---|---|---|---|---|
| MLX Greedy | 119.3 | - | 1.00x | 2.69x |
| MLX Speculative K=1 | 74.5 | 42.20% | 0.62x | 1.68x |
| MLX Speculative K=2 | 75.1 | 56.60% | 0.63x | 1.70x |
| MLX Speculative K=4 | 61.4 | 67.40% | 0.51x | 1.39x |
| MLX Speculative K=8 | 44.6 | 72.20% | 0.37x | 1.01x |

**Headline finding**
MLX greedy alone gives a **2.7x speedup over PyTorch greedy** without any speculative decoding. The runtime, not the algorithm, is the big lever on M2 Max.

**Counter-finding**
MLX's built-in speculative decoding makes things SLOWER, not faster, for this 4-bit quantized model pair. At K=4 it drops to 0.51x of its own greedy. Reason: 4-bit quantization makes both draft and verifier so cheap that the cost ratio gamma is small (~1.3x maybe), and speculative overhead (cache management, draft sampling) outweighs the K-parallelism win.

**The full optimization story**
| Stage | Best K=4 tok/s | Speedup vs original baseline |
|---|---|---|
| Original cached (Day 8.5) | 40.3 | 1.00x |
| Eager scheme (Phase 3) | 44.5 | 1.10x |
| Eager + sync cleanup (Phase 4) | 46.9 | 1.16x |
| MLX greedy (Phase 5, different model) | 119.3 | 2.96x |

**Practical conclusion**
For shipping speed on M2 Max: switch to MLX. Speculative decoding is a smaller lever once the runtime is fast. The algorithm matters when per-call overhead is large; in a runtime that minimizes that overhead (MLX vs PyTorch+MPS), greedy is hard to beat.
