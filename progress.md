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
**Status**: Not started

## Day 3 - Draft Model: Propose K Tokens
**Status**: Not started

## Day 4 - Verifier: Score Draft Sequence
**Status**: Not started

## Day 5 - End-to-End Speculative Decoding Loop
**Status**: Not started

## Day 6 - Measure Acceptance Rate
**Status**: Not started

## Day 7 - Throughput Benchmark
**Status**: Not started

## Day 8 - Sequence Length Sweep
**Status**: Not started

## Day 9 - Profiling
**Status**: Not started

## Day 10 - Clean README + Final Benchmark Table
**Status**: Not started
