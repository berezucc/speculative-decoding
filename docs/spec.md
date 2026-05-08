# Speculative Decoding — Project Spec

## What You're Building

A from-scratch speculative decoding implementation that runs on Apple Silicon (MPS/CPU).
Benchmark it against greedy baseline and produce real numbers.

**One-liner for LinkedIn**: *"Implemented speculative decoding from first principles in PyTorch — 2x+ throughput over greedy baseline on Apple M2 Max, with acceptance rate and speedup analysis across 3 temperature regimes."*

---

## The Problem Speculative Decoding Solves

Autoregressive generation is serial: each token requires one full forward pass of the large model.
On modern hardware, transformer generation is **memory-bandwidth bound**, not compute bound.
The GPU (or ANE) is mostly idle waiting on weight reads, not doing math.

The insight: if you could verify K candidate tokens in one parallel forward pass instead of generating them one by one, you'd get K tokens for the cost of roughly 1 large model pass.

---

## Core Concepts to Understand Before Coding

### 1. Why generation is memory-bandwidth bound
Each forward pass during autoregressive decoding reads all model weights (~500MB for gpt2-medium)
to produce a single token. The arithmetic intensity is very low — lots of memory reads, little compute.
This is the fundamental inefficiency that speculative decoding exploits.

Read: https://finbarr.ca/how-is-llama-cpp-so-fast/ (10 min, gives the intuition clearly)

### 2. KV cache
During generation, attention keys and values from previous tokens don't change.
You cache them and only compute the new token's K/V. Without this, generation is O(n²).
You must understand KV cache before speculative decoding — the draft model and verifier both maintain separate caches.

Read: The Illustrated GPT-2 (Jay Alammar) — the generation section specifically.

### 3. The acceptance criterion (the actual math)

Let:
- `p(x)` = probability the verifier assigns to token x at position i
- `q(x)` = probability the draft model assigns to token x at position i

Accept the draft token with probability `min(1, p(x_i) / q(x_i))`.

If rejected, sample from the *residual distribution*:
```
p_adjusted(x) = max(0, p(x) - q(x))
p_adjusted normalized
```

This guarantees the output distribution matches sampling from the verifier alone.
This is the key correctness guarantee — speculative decoding is lossless (same distribution as pure verifier).

### 4. Expected speedup

Let:
- `K` = number of draft tokens proposed per step
- `α` = mean acceptance rate (0-1)
- `γ` = cost ratio: verifier / draft (how many times faster the draft model is)

Expected tokens per decoding step: `1 + K * α` (the 1 is the guaranteed bonus token from the adjusted dist)

Effective speedup ≈ `(1 + K * α) / (1 + K / γ)`

For distilgpt2 (draft) + gpt2-medium (verifier):
- `γ ≈ 3-4x` (distilgpt2 is roughly 3-4x faster)
- `α ≈ 0.6-0.8` at low temperature (higher temp = lower acceptance)
- `K = 4` → expected speedup ≈ 1.8-2.2x

### 5. Why draft and verifier must share tokenizer
The acceptance criterion compares `p(x_i)` and `q(x_i)` for the same token id `x_i`.
If the models have different vocabularies, this comparison is undefined.
distilgpt2 and gpt2-medium both use gpt2's BPE tokenizer — same 50257-token vocab.

---

## Papers to Read (in order, before or while coding)

| Priority | Paper | Why |
|---|---|---|
| Required | [Leviathan et al. 2023 — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | The original paper. Read the algorithm section (Section 2) carefully. |
| Required | [Chen et al. 2023 — Speculative Sampling (DeepMind)](https://arxiv.org/abs/2302.01318) | Same idea, independent discovery. Different framing, clearer math in places. |
| Optional | [Cai et al. 2024 — MEDUSA](https://arxiv.org/abs/2401.10774) | Multiple draft heads instead of a separate model. Simpler in practice. |
| Optional | [Miao et al. 2024 — SpecInfer](https://arxiv.org/abs/2305.09781) | Tree-based speculative decoding — proposes a tree of candidates, not a single sequence. |

Read both required papers before Day 3. The algorithm is ~2 pages each.

---

## Project Structure

```
speculative-decoding/
├── src/
│   ├── baseline.py          # greedy + temperature sampling, no speculative
│   ├── draft.py             # draft model wrapper: propose K tokens
│   ├── verifier.py          # verifier model wrapper: score token sequences
│   ├── speculative.py       # the speculative decoding loop
│   ├── benchmark.py         # timing harness, tokens/sec measurement
│   └── utils.py             # model loading, tokenizer helpers, seed setting
├── notebooks/
│   └── analysis.ipynb       # acceptance rate curves, speedup charts
├── benchmarks/
│   └── results.md           # benchmark tables (filled in during Week 2)
├── docs/
│   └── spec.md              # this file
├── requirements.txt
└── README.md                # filled in on Day 10
```

---

## Sprint Plan

### Week 1: Working Implementation

**Day 1 — Baseline greedy decoding**
- Load distilgpt2 and gpt2-medium with HuggingFace
- Write a simple autoregressive loop (no `model.generate()` — do the forward pass yourself)
- Time it: tokens/sec for 100 tokens, prompt length 32 and 128
- Understand the shape of `logits`, what `past_key_values` looks like
- Commit: `baseline: X tok/s on distilgpt2, Y tok/s on gpt2-medium`

Goal: before Day 2, you can call `model(input_ids, past_key_values=...)` yourself and get next-token logits.

**Day 2 — Read the papers + understand the math**
- Read Leviathan et al. Section 1-3 (algorithm + correctness proof sketch)
- Read Chen et al. for alternate framing
- Implement the acceptance criterion as a standalone function you can unit-test:
  ```python
  def acceptance_prob(p_logits, q_logits, token_id) -> float:
      ...
  def sample_residual(p_logits, q_logits) -> int:
      ...
  ```
- Test: if p == q, acceptance rate should be 1.0. If p >> q for a token, it should always accept.
- Commit: `acceptance criterion implemented and tested`

**Day 3 — Draft model: propose K tokens**
- Write `DraftModel.propose(input_ids, k) -> (draft_tokens, draft_probs)`:
  - Run the draft model autoregressively for K steps
  - Store the probability assigned to each chosen token
  - Keep the KV cache to avoid recomputation
- Test with K=4: given a prompt, get 4 proposed tokens and their probs
- Commit: `draft model proposing K tokens with cached KV`

**Day 4 — Verifier: score a draft sequence**
- Write `VerifierModel.score(input_ids, draft_tokens) -> verifier_probs`:
  - Feed `[input_ids | draft_tokens]` to the verifier in ONE forward pass
  - Extract the verifier's probability for each draft token position
  - This is the key efficiency: one pass scores K tokens in parallel
- Verify shape: should return K probabilities (one per draft token position)
- Commit: `verifier scoring draft sequence in one forward pass`

**Day 5 — End-to-end speculative decoding loop**
- Combine draft + verifier + acceptance criterion into `speculative_generate()`
- Handle the rejection case: when token i is rejected, resample from residual and stop (tokens i+1..K are discarded)
- Handle the acceptance case: all K accepted, generate one bonus token from verifier's last logits
- Verify correctness: run greedy (temperature=0) and check output matches verifier-only greedy
- Commit: `speculative decoding working end-to-end, correctness verified`

---

### Week 2: Benchmarking + Polish

**Day 6 — Measure acceptance rate**
- Run across 20 different prompts (varied domains: code, prose, factual, creative)
- Vary temperature: 0.0, 0.5, 1.0, 1.5
- Vary K: 1, 2, 4, 8
- Log acceptance rate α for each combination
- Finding: what conditions maximize α? Where does it fall below 0.5?
- Commit: `acceptance rate analysis — α table`

**Day 7 — Throughput benchmark**
- Compare on 100-token generation, 5 prompts each:
  - Greedy (gpt2-medium only)
  - Speculative K=2, K=4, K=8 (distilgpt2 + gpt2-medium)
- Measure: tokens/sec, wall time
- Calculate: observed speedup vs theoretical prediction from Day 6's α
- Commit: `throughput benchmark table`

**Day 8 — Sequence length sweep**
- How does speedup change as the prompt grows? (32, 128, 512 tokens)
- KV cache grows → memory pressure → does acceptance rate change?
- Commit: `sequence length benchmark`

**Day 9 — Profile where time goes**
- Use `torch.profiler` or manual timing to break down:
  - Draft model time per step
  - Verifier scoring time
  - Acceptance logic time (should be negligible)
- Answer: is the bottleneck the draft model, the verifier, or memory transfers?
- On M2 Max: check if MPS is faster than CPU for each model
- Commit: `profiling breakdown`

**Day 10 — Clean README + benchmark table**
- Write README: what is speculative decoding, how the code works, how to run it
- Add benchmark table with: model pair, K, temperature, α, speedup
- Add one diagram: the accept/reject loop
- Write 3 sentences: what surprised you, what the bottleneck was, when speculative decoding helps vs hurts
- Commit: `final clean repo`

---

## Key Metrics to Track

| Metric | What it measures |
|---|---|
| tokens/sec (baseline) | greedy throughput, your reference number |
| tokens/sec (speculative) | throughput with speculative decoding |
| speedup | speculative / baseline ratio |
| acceptance rate α | fraction of draft tokens accepted |
| theoretical speedup | (1 + K*α) / (1 + K/γ) — how close to theory? |
| memory (peak) | both models loaded simultaneously |

---

## Correctness Tests (before benchmarking)

At temperature=0 (greedy), speculative decoding must produce identical output to greedy verifier-only decoding.
If it doesn't, you have a bug in the acceptance logic or KV cache management.

```python
def test_correctness():
    greedy_output = greedy_generate(verifier, prompt, n=50)
    spec_output = speculative_generate(draft, verifier, prompt, n=50, temperature=0)
    assert greedy_output == spec_output, "Output mismatch — check acceptance logic"
```

Run this before you trust any benchmark numbers.

---

## What Can Go Wrong

- **KV cache shape mismatch**: draft and verifier have different n_head / n_layer. Keep caches separate and never mix them.
- **Off-by-one in token positions**: when scoring the draft, you want the logits at positions [i, i+1, ..., i+K-1] — be careful which logit corresponds to which token.
- **Numerical issues in residual sampling**: `max(0, p - q)` can underflow. Add a small epsilon or use log-space.
- **Temperature applied inconsistently**: apply temperature in the same way to both draft and verifier, or acceptance rates won't match theory.
