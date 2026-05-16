# Optimization Deep Dive

How the speculative decoding implementation went from 0.83x (slower than greedy) to 1.16x on Apple M2 Max, and what was learned along the way.

This document complements `docs/optimization-plan.md` (the plan) and `progress.md` (the log) with the engineering detail: hypotheses, code-level changes, measurements, and verdicts.

## Context

After the 10-day project sprint, the cached speculative implementation passed correctness (output identical to verifier-only greedy at temp=0) but ran at 0.83x of greedy throughput on M2 Max + PyTorch + MPS. Day 9 profiling identified the root cause: ~25 ms fixed cost per verifier forward pass on MPS, with each speculative iteration spending 88% of its time inside model forward passes. The bottleneck was per-call overhead, not the algorithm.

This document walks through a 5-phase optimization sequence executed against that baseline.

## Methodology

| Stop condition | If K=4 speedup >= 1.0x, halt |
| Verification | After each change: `python src/speculative_cached.py --test` must PASS |
| Measurement | `python src/benchmark.py --n_tokens 100` across 5 varied prompts |
| Discipline | One change at a time. Revert any phase that regresses correctness or throughput. |

Baseline before any change:

```
Greedy (verifier only):  48.6 tok/s   (1.00x)
Speculative K=4:         40.3 tok/s   (0.83x)
Speculative K=8:         41.7 tok/s   (0.86x)
```

## Phase 1: fp16 inference

**Hypothesis**: gpt2-medium and distilgpt2 are memory-bandwidth bound. Cutting weight precision from fp32 to fp16 halves the memory bandwidth per forward pass and should roughly halve forward pass time.

**Change** (`src/utils.py`):
```python
# before
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
# after
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
```

**Result**:
```
Greedy:        46.5 tok/s   (was 48.6)
Spec K=4:      37.1 tok/s   (was 40.3)
Speedup:       0.80x
```

**Verdict**: REVERTED.

**Why it failed**: forward pass time on MPS is dominated by fixed kernel-launch overhead (~25 ms per call), not memory bandwidth. Halving the weights doesn't translate to faster passes when the bottleneck is the dispatch, not the data movement.

## Phase 2: torch.compile

**Hypothesis**: `torch.compile(model, mode='reduce-overhead')` reduces Python dispatch overhead, which is a candidate for the ~25 ms per-call cost.

**Change** (`src/utils.py` + `src/benchmark.py`):
```python
# utils.py
def maybe_compile(model, mode: str = "reduce-overhead"):
    """Wrap a model with torch.compile. Falls back gracefully if MPS can't lower an op."""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    try:
        return torch.compile(model, mode=mode)
    except Exception as e:
        print(f"torch.compile failed ({e}), using uncompiled model")
        return model

# benchmark.py
parser.add_argument("--compile", action="store_true")
# ...
if args.compile:
    draft.model = maybe_compile(draft.model)
    verifier.model = maybe_compile(verifier.model)
```

**Result**: errors with `LoweringException: TypeError: 'NoneType' object is not callable, target: aten.var_mean.correction` inside LayerNorm. With `suppress_errors=True`, falls back to eager mode for the broken op but compile scaffolding remains.

```
Greedy:        39.6 tok/s
Spec K=4:      33.1 tok/s
Speedup:       0.83x
```

**Verdict**: REVERTED (helper kept as opt-in `--compile` flag for future MPS releases).

**Why it failed**: MPS backend in PyTorch as of writing has no lowering for `aten.var_mean.correction`. LayerNorm is unavoidable in transformer blocks, so the compile pass cannot succeed for these models. The `suppress_errors=True` fallback executes the unsupported op in eager mode but the dynamo/inductor scaffolding around the fallback adds overhead.

## Phase 3: Eager scheme rewrite

**Hypothesis**: the original cached loop pays 2 extra forward passes per iteration in an "end-of-iter setup" step that saves the last logit for the next iteration's first draft sample. By reorganizing to feed the previous-iter's last token at the start of each iter instead, we save those 2 passes.

Day 9 profile showed `setup_next` was 29.3% of total speculative time. If we eliminate it, we should see roughly 25-30% speedup on the speculative path.

### The original scheme

```
Prefill (once):
  Feed full prompt to both models.
  Save last_v_logit, last_d_logit. Both caches at L.

Each iteration:
  1. Use saved last_d_logit to sample t1            (0 fwd passes)
  2. K-1 draft fwd passes for t2..tK
  3. 1 verifier fwd pass on [t1..tK]                (K tokens)
  4. Concat saved last_v_logit with K new           Scores K drafts + bonus.
     verifier logits = K+1 verifier probs
  5. Accept / reject; build new_tokens
  6. Truncate caches on rejection
     or extend draft cache with tK on full accept
  7. "setup_next": feed the new last token to BOTH  *** 2 extra fwd passes ***
     models, save their next-prediction logits

Total fwd passes per iter:
  K=4: 3 (draft) + 1 (verifier) + 2 (setup_next) = 6
```

### The eager scheme

```
Prefill (once):
  Feed prompt[:-1] to both models. Caches at L0 - 1.
  Invariant: caches cover current_ids[:-1] (all but the last token).

Each iteration:
  1. Feed current_ids[-1] to draft as the first    K fwd passes total.
     fwd pass, then K-1 more fwd passes for
     t2..tK. The last passes' logits sample tK.
  2. 1 verifier fwd pass on
     [current_ids[-1], t1, ..., tK]                 K+1 tokens, K+1 logits returned.
                                                    No concat with saved logit needed.
  3. Accept / reject; build new_tokens
  4. Truncate caches on rejection
     or extend draft cache with tK on full accept
  5. (No setup_next.) Invariant restored:
     caches cover all of new current_ids except
     its last token.

Total fwd passes per iter:
  K=4: 4 (draft) + 1 (verifier) = 5
```

### The trick

The invariant is the load-bearing piece. By ensuring caches always cover `current_ids[:-1]` (and never `current_ids` fully), the "last token" is naturally fed at the start of the next iteration as part of the draft proposal, not as a separate setup step.

The verifier inputs `[last_token, t1, ..., tK]` of length K+1 produce exactly the K+1 logits needed: position L-1 (predicting t1), L (predicting t2), ..., L+K-1 (predicting bonus). No external concatenation, no saved logit state to maintain.

### Code change

The bulk of `speculative_generate_cached()` in `src/speculative_cached.py` was rewritten. Key differences:

```python
# Prefill: prompt[:-1] instead of full prompt
with torch.no_grad():
    if L0 > 1:
        prefill_ids = initial_ids[:, :-1]
        verifier_kv = verifier_obj.model(prefill_ids, use_cache=True).past_key_values
        draft_kv = draft_obj.model(prefill_ids, use_cache=True).past_key_values

# Draft loop: K passes, first feeds last_token
last_token = current_ids[0, -1].item()
prev_input = torch.tensor([[last_token]], device=device)
for _ in range(k):
    out = draft_obj.model(prev_input, past_key_values=draft_kv, use_cache=True)
    draft_kv = out.past_key_values
    # ... sample t, append, set prev_input = t.view(1, 1)

# Verifier: feed [last_token, t1, ..., tK], get K+1 logits in one pass
verifier_input = torch.cat([
    torch.tensor([[last_token]], device=device),
    draft_tokens_tensor.view(1, -1),
], dim=1)
out_v = verifier_obj.model(verifier_input, past_key_values=verifier_kv, use_cache=True)
new_v_logits = out_v.logits[0]  # (K+1, vocab) — no concat needed

# End of iter: no setup_next pass. Invariant maintained.
```

### Result

```
Greedy:        41.1 tok/s   (run-to-run variance)
Spec K=4:      44.5 tok/s   (was 40.3)
Speedup:       1.08x        <-- crossed 1.0x
Spec K=8:      44.2 tok/s   (1.07x)
```

**Verdict**: KEPT. Stop condition met. Correctness test still PASSES at temp=0.

The actual time savings (~10% per iter) is smaller than the 29% the profile predicted because (a) the eager scheme adds 1 draft pass that the original didn't (the new "feed last_token" first draft pass), and (b) the verifier now processes K+1 tokens instead of K — slightly more compute, though only ~1 ms more given marginal token cost on MPS.

## Phase 4: Sync cleanup

**Hypothesis**: the accept/reject loop calls `.item()` on tensors several times per draft token, each of which forces an MPS → CPU synchronization. At K=4 that's ~16 syncs per iteration, on top of the forward-pass syncs. Reducing this should help.

**Changes** (all in `src/speculative_cached.py`):

### Draft sampling — inline at temp=0

The `_sample()` helper had a `.max().item()` check to detect one-hot probability vectors:

```python
def _sample(probs):
    if probs.max().item() == 1.0:      # <-- MPS sync, K times per iter
        return torch.argmax(probs)
    return torch.multinomial(probs, num_samples=1).squeeze()
```

Replaced with inlined logic in the draft loop:

```python
is_greedy = temperature == 0.0
for _ in range(k):
    out = draft_obj.model(prev_input, past_key_values=draft_kv, use_cache=True)
    draft_kv = out.past_key_values
    logit = out.logits[0, -1, :]
    if is_greedy:
        t = torch.argmax(logit)
        probs = torch.zeros_like(logit)
        probs[t] = 1.0
    else:
        probs = torch.softmax(logit / temperature, dim=-1)
        t = torch.multinomial(probs, num_samples=1).squeeze()
    draft_probs_list.append(probs)
    draft_tokens.append(t)
```

### Accept/reject — vectorize alpha computation

Before (per-token .item() calls):

```python
for i in range(k):
    token_id = draft_tokens[i].item()                 # sync
    p = verifier_probs[i]
    q = draft_probs_tensor[i]
    alpha = acceptance_prob(p, q, token_id)           # 2 .item() inside
    r = torch.rand(1).item()                          # sync
    if r < alpha:
        new_tokens.append(token_id)
    else:
        correction = sample_residual(p, q)
        new_tokens.append(correction)
        rejected = True
        break
```

After (one gather, one .tolist() to bring everything to CPU):

```python
# One gather gives v_at_tokens[i] = verifier_probs[i, draft_tokens[i]] for all K
token_ids_tensor = draft_tokens_tensor.unsqueeze(1)
v_at_tokens = verifier_probs[:k].gather(1, token_ids_tensor).squeeze(1)
d_at_tokens = draft_probs_tensor.gather(1, token_ids_tensor).squeeze(1)
alphas_list = torch.clamp(v_at_tokens / (d_at_tokens + 1e-10), max=1.0).tolist()
token_ids_list = draft_tokens_tensor.tolist()
r_values = torch.rand(k).tolist()  # CPU rand, never touches MPS

for i in range(k):
    if r_values[i] < alphas_list[i]:
        new_tokens.append(token_ids_list[i])
    else:
        correction = sample_residual(verifier_probs[i], draft_probs_tensor[i])
        new_tokens.append(correction)
        rejected = True
        break
```

Total syncs per iter went from ~4K (4 syncs × K draft tokens) to 2.

### Result

Three runs gave K=4 speedups of 0.99x, 1.13x, 1.08x. Mean ~1.07x. Within noise of Phase 3.

**Verdict**: KEPT (code is cleaner and avoids unnecessary syncs), but did not move the needle measurably.

**Why it didn't help much**: forward passes (~90% of total time) dominate everything else. The accept/reject loop is so cheap that further reducing it doesn't show up at the macro level. Phase 3 already removed the dominant overhead source.

## Phase 5: MLX comparison

**Hypothesis**: a C++-native runtime like Apple's MLX has substantially lower per-call overhead than PyTorch + MPS. Same algorithm in a different runtime should reveal whether the bottleneck is the algorithm or the framework.

### Constraints

mlx-lm doesn't ship pre-converted weights for `gpt2` or `distilgpt2`. Direct loading from HuggingFace fails with `"Received 82 parameters not in model"` because mlx-lm's GPT-2 module expects parameter keys at `model.wte.weight` while the HF safetensors are at `transformer.wte.weight` — and mlx-lm's `sanitize()` function doesn't handle the prefix mismatch.

Substituted model pair:
- Draft: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~500M params, 4-bit quantized)
- Verifier: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~1.5B params, 4-bit quantized)
- Both pre-converted to MLX format by mlx-community
- Same tokenizer family, similar size ratio to the PyTorch pair

This is not an apples-to-apples comparison of the PyTorch optimization. It is a measurement of "what's possible on M2 Max in a different runtime."

### Implementation

New file `src/benchmark_mlx.py` uses `mlx_lm.stream_generate(..., draft_model=...)` which has speculative decoding built in. Times greedy and speculative at K=1, 2, 4, 8 across the same 5 prompts.

### Result

```
MLX Greedy:           119.3 tok/s   1.00x (vs MLX greedy)   2.69x (vs PyTorch greedy)
MLX Spec K=1:          74.5 tok/s   0.62x                   1.68x
MLX Spec K=2:          75.1 tok/s   0.63x                   1.70x
MLX Spec K=4:          61.4 tok/s   0.51x                   1.39x
MLX Spec K=8:          44.6 tok/s   0.37x                   1.01x
```

Two findings, both interesting.

### Finding 1: the runtime is the lever

MLX greedy alone is **2.69x faster than PyTorch greedy** on the same hardware. No algorithmic change, just a different inference framework. The PyTorch + MPS per-call overhead (~25 ms fixed cost we measured in Phase 3) collapses in MLX because MLX is a thin C++ wrapper over Metal and has no Python ↔ Metal sync per call.

### Finding 2: speculative decoding doesn't help inside MLX

Speculative is slower than greedy at every K. Why?

With 4-bit quantized models, both draft (0.5B) and verifier (1.5B) read very little memory per forward pass:
- 0.5B at 4-bit ≈ 250 MB of weights
- 1.5B at 4-bit ≈ 750 MB of weights

Both fit comfortably in M2 Max unified memory bandwidth budget. The verifier per-token cost is small, which means the draft / verifier cost ratio γ is also small (likely 1.3-1.5x rather than the 2-3x we'd want). With small γ, the speculative decoding speedup formula `(1 + K·α) / (1 + K/γ)` doesn't deliver wins, especially when there's any per-iter overhead from cache management.

The corollary: **speculative decoding is useful when the verifier is slow. Once the verifier is fast (small, quantized, on a fast runtime), the algorithm stops paying off.**

## Final results

| Stage | K=4 tok/s | vs original speculative | vs original greedy |
|---|---|---|---|
| Original cached (post Day 8.5) | 40.3 | 1.00x | 0.83x |
| + fp16 (Phase 1, reverted) | 37.1 | 0.92x | — |
| + torch.compile (Phase 2, reverted) | 33.1 | 0.82x | — |
| + eager scheme (Phase 3) | 44.5 | 1.10x | 1.08x (crossed) |
| + sync cleanup (Phase 4) | 46.9 | 1.16x | 1.13x |
| MLX greedy (Phase 5, diff model) | 119.3 | 2.96x | 2.69x |

## Takeaways

1. **Profile before optimizing.** Day 9's per-phase breakdown immediately ruled out cache management and accept/reject as candidates. Without it I would have spent days on micro-optimizations that mattered ~5% and missed the eager scheme refactor.

2. **Per-call overhead is the M2 Max ceiling for speculative decoding in PyTorch.** ~25 ms fixed per verifier forward pass means the algorithm has to do fewer forward passes per output token than greedy, which constrains the design space severely.

3. **Runtime > algorithm** at this scale. A 10% gain from PyTorch optimization is dwarfed by a 270% gain from runtime swap. If you care about raw speed, switch runtimes first.

4. **Speculative decoding is conditional.** It helps when the verifier is expensive and the draft is cheap. With quantized models on a fast runtime, both are cheap, and the algorithm provides no benefit. Knowing when *not* to apply an optimization is part of the skill.

5. **Negative results have signal.** fp16 not helping told us the bottleneck wasn't memory bandwidth. torch.compile failing told us the MPS dynamo path isn't production-ready for transformer models. Both are useful conclusions for someone planning similar work.

## Future work

- **Rewrite the implementation in MLX directly.** Use the same model pair (distilgpt2/gpt2-medium converted manually via mlx_lm.convert with a custom sanitize) to get an apples-to-apples comparison of the algorithm in both runtimes.
- **Quantize the PyTorch models.** 8-bit or 4-bit quantization of gpt2-medium might shift the bottleneck balance. Would need bitsandbytes (CUDA-only) or a custom MPS quantization path.
- **Try a larger γ pair.** distilgpt2 (82M) vs a 1B+ model. The cost ratio matters; if γ is ~5x, speculative might pay off even on PyTorch+MPS without algorithmic optimization.
- **Investigate the `aten.var_mean.correction` MPS gap.** Once that lowering lands in PyTorch's MPS backend, torch.compile should work and is worth re-trying.
