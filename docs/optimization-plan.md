# Optimization Plan: M2 Max Speculative Decoding

## Goal
Push the cached speculative implementation above 1.0x speedup vs greedy baseline on M2 Max + PyTorch + MPS.

## Current baseline (from Day 7)
- Greedy: 48.6 tok/s
- Speculative K=4: 40.3 tok/s, 0.83x
- Speculative K=8: 41.7 tok/s, 0.86x
- Root cause: ~25 ms fixed cost per verifier forward pass, ~9 ms per draft pass

## How to run each benchmark
After each change, run these three commands in order:

```bash
python src/speculative_cached.py --test            # correctness must still PASS
python src/benchmark.py --n_tokens 100             # measures throughput vs greedy
python src/profile_breakdown.py                    # confirms where time went
```

Record numbers in this doc under the relevant phase.

## Stop conditions
- Any phase: if K=4 speedup crosses 1.0x, you can stop (project done)
- Any phase: if correctness test fails, revert and move to next phase
- After Phase 4: if nothing crossed 1.0x, MLX (Phase 5) is the only remaining lever

---

## Phase 1: fp16 inference

**Hypothesis**: gpt2-medium and distilgpt2 are memory-bandwidth bound. Cutting weight size in half should roughly halve the forward pass time for the memory-bound portion.

**Change**: in `src/utils.py`:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
```

**Run**:
```bash
python src/speculative_cached.py --test
python src/baseline.py                             # see fp16 greedy baseline
python src/benchmark.py --n_tokens 100
```

**Expected outcomes**:
- Greedy baseline likely jumps from 48 tok/s to 70-90 tok/s
- Speculative tok/s also rises proportionally
- Speedup ratio may or may not improve (both halves rise together)
- If MPS fp16 numerics break correctness, switch to bf16 or skip phase

**Record results below**:
- Greedy fp16: ___ tok/s
- Speculative K=4: ___ tok/s, ___x speedup
- Correctness: PASS / FAIL

**Decision**: if speedup >= 1.0x at any K, stop. Otherwise proceed.

---

## Phase 2: torch.compile

**Hypothesis**: `mode='reduce-overhead'` reduces per-call Python and dispatch overhead, which dominates on MPS.

**Change**: in the benchmark scripts (or wrap right after `load_model`):
```python
import torch
draft.model = torch.compile(draft.model, mode='reduce-overhead')
verifier.model = torch.compile(verifier.model, mode='reduce-overhead')
```

**Run**:
```bash
python src/speculative_cached.py --test
python src/benchmark.py --n_tokens 100
```

**Expected outcomes**:
- First call slow (compilation), subsequent fast
- If MPS support fails, will error or silently fall back to eager. Try `mode='default'` as fallback. If still broken, skip.
- If it works, could be 2-3x. Could also do nothing.

**Record results below**:
- Greedy with compile: ___ tok/s
- Speculative K=4: ___ tok/s, ___x speedup
- Correctness: PASS / FAIL / ERRORED

**Decision**: if speedup >= 1.0x at any K, stop. Otherwise proceed.

---

## Phase 3: Eager scheme rewrite

**Hypothesis**: removing the `setup_next` phase (29.3% of speculative time per Day 9 profile) saves 1 forward pass per iter, ~15% faster.

**Change**: rewrite `src/speculative_cached.py`:
- Remove the end-of-iter `last_v_logit` and `last_d_logit` saves
- Remove the end-of-iter forward passes that produced them
- At the start of each iteration, feed `current_ids[-1]` (the last new token from previous iter) to the draft as part of the K-step proposal loop, getting K draft tokens via K forward passes (not K-1)
- For the verifier, feed `[current_ids[-1], t1, ..., t(K-1)]` as one K-token pass to score t1..tK, then feed tK separately to extend the cache (or include it in the same pass: `[current_ids[-1], t1, ..., tK]` as K+1 tokens, getting K+1 logits in one go)
- On first iter (just after prefill), there is no previous-iter last token, but there is the last token of the prompt; logic stays the same

This trades:
- Saved: 2 end-of-iter forward passes per iter (1 verifier, 1 draft)
- Added: K-th draft proposal pass (we had K-1, now K) and the K+1th verifier token in the score pass
- Net: ~1 fewer forward pass per iter

**Run**:
```bash
python src/speculative_cached.py --test
python src/benchmark.py --n_tokens 100
python src/profile_breakdown.py                    # should show setup_next gone
```

**Record results below**:
- Speculative K=4: ___ tok/s, ___x speedup
- setup_next % of total in profile: should be 0%
- Correctness: PASS / FAIL

**Decision**: if speedup >= 1.0x at any K, stop. Otherwise proceed.

---

## Phase 4: Reduce CPU/GPU sync

**Hypothesis**: every `.item()` call in the inner loop forces an MPS sync. Batching them saves time.

**Changes** in `speculative_cached.py`:
1. Pre-roll all K random numbers once: `r_values = torch.rand(k).tolist()` outside the accept loop
2. Compute alpha for all K in one shot before the loop: `alphas = torch.clamp(p / (q + 1e-10), max=1.0)` then convert to list once
3. Avoid `.clone()` on saved logits unless required for correctness
4. Audit every `.item()` call in `acceptance.py` and `speculative_cached.py`

**Run**:
```bash
python src/speculative_cached.py --test
python src/benchmark.py --n_tokens 100
```

**Record results below**:
- Speculative K=4: ___ tok/s, ___x speedup
- Correctness: PASS / FAIL

**Decision**: if speedup >= 1.0x at any K, stop. Otherwise proceed.

---

## Phase 5: MLX port (if all of 1-4 fail)

**When to do this**: only if Phases 1-4 combined still leave you below 1.0x.

**Scope**: separate implementation in MLX, not a modification of the PyTorch code. New directory `mlx_impl/`:
- `mlx_impl/draft.py`
- `mlx_impl/verifier.py`
- `mlx_impl/speculative.py`
- `mlx_impl/benchmark.py`

**Compare against**:
- Your PyTorch+MPS numbers from Phases 1-4
- `mlx-lm`'s built-in speculative decoding (it ships with one)

**Effort**: 2-3 days.

**Deliverable**: a table comparing PyTorch+MPS (optimized), MLX (yours), MLX (mlx-lm built-in). This becomes the project's strongest finding.

---

## Execution rules

1. **Do phases in order**. Skipping phase N means you don't know what it would have given you.
2. **Always run correctness test first** after a change. If it fails, revert and either fix or move on.
3. **Record numbers in this doc** as you go. Don't trust memory.
4. **One change at a time**. If you stack changes and something breaks, you won't know which one.
5. **If a phase produces a regression**, revert it and continue to the next phase from the previous good state.

---

## Summary table (fill in as you go)

| Phase | Change | Greedy tok/s | Spec K=4 tok/s | Speedup | Correctness |
|---|---|---|---|---|---|
| 0 (baseline) | fp32, current code | 48.6 | 40.3 | 0.83x | PASS |
| 1 | fp16 | 46.5 | 37.1 | 0.80x | PASS (reverted: no benefit, MPS per-call overhead dominates math savings) |
| 2 | torch.compile | 39.6 | 33.1 | 0.83x | PARTIAL (MPS cannot lower aten.var_mean.correction in LayerNorm; suppress_errors falls back to eager with overhead) |
| 3 | eager scheme | 41.1 | 44.5 | **1.08x** | PASS (K=8: 1.07x; crossed 1.0x, stop condition met) |
| 4 | sync cleanup | 43.4 | 46.9 | 1.08x (median of 3) | PASS (kept, neutral vs Phase 3 within noise; forward passes dominate) |
| 5 | MLX (Qwen 0.5B/1.5B 4-bit) | 119.3 | 61.4 | 0.51x (within MLX); MLX-greedy alone is 2.7x of PyTorch-greedy | PASS (correctness via output coherence; MLX runtime gives huge speedup over PyTorch but speculative no longer helps once greedy is already memory-bandwidth-fast) |
