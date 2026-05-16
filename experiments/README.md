# Experiments

Historical / exploratory code from the project. Kept for reference and reproducibility, but **not** part of the canonical implementation in `src/`.

## Files

| File | Purpose |
|---|---|
| `speculative_naive.py` | The original speculative decoding loop without KV cache persistence. Used during Days 1-7 of the project. Exposed in Day 8's sequence length sweep as having an O(n × m) inefficiency (reprocesses the full prompt every iteration). Replaced by `src/speculative.py`. Useful as a teaching artifact for "what NOT to do." |

## Why this is separate from `src/`

`src/` is the canonical, production implementation. This directory contains code that was useful for learning or that documents what was tried and rejected. Code here may be slower, buggier, or inconsistent with the current API. Do not use it in benchmarks.
