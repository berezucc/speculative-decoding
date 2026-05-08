# Speculative Decoding

From-scratch implementation of speculative decoding for LLM inference, benchmarked on Apple M2 Max.

**Status**: In progress

---

## What is speculative decoding?

Autoregressive LLM generation is serial: each token requires one full forward pass of the large model. For a 355M parameter model, that means reading ~700MB of weights per token — which is memory-bandwidth bound, not compute bound.

Speculative decoding breaks this bottleneck:
1. A small **draft model** proposes K tokens cheaply (fast, low quality)
2. The large **verifier model** scores all K tokens in a single parallel forward pass
3. Tokens are accepted or rejected based on the probability ratio `p(x) / q(x)`
4. The output distribution is **identical** to pure verifier sampling (lossless)

Result: ~2x throughput on matched model pairs, with no change in output quality.

---

## Model pair

| Role | Model | Parameters |
|---|---|---|
| Draft | distilgpt2 | 82M |
| Verifier | gpt2-medium | 355M |

Both use the same BPE tokenizer (50257 vocab) — required for the acceptance criterion.

---

## Benchmark Results

*(filled in at project completion)*

| Method | K | Temperature | Acceptance rate α | tokens/sec | Speedup |
|---|---|---|---|---|---|
| Greedy (verifier only) | — | 0.0 | — | — | 1.0x |
| Speculative | 4 | 0.0 | — | — | —x |
| Speculative | 4 | 0.5 | — | — | —x |
| Speculative | 8 | 0.0 | — | — | —x |

---

## How to run

```bash
pip install -r requirements.txt

# Baseline greedy decoding
python src/baseline.py --prompt "The transformer architecture" --n_tokens 100

# Speculative decoding
python src/speculative.py --prompt "The transformer architecture" --n_tokens 100 --k 4

# Full benchmark
python src/benchmark.py
```

---

## Project structure

```
src/
  baseline.py      # greedy decoding loop (no model.generate())
  draft.py         # draft model: propose K tokens with KV cache
  verifier.py      # verifier model: score draft sequence in one pass
  speculative.py   # the speculative decoding loop
  benchmark.py     # timing harness
  utils.py         # model loading, tokenizer, seed helpers
notebooks/
  analysis.ipynb   # acceptance rate curves, speedup charts
benchmarks/
  results.md       # benchmark tables
docs/
  spec.md          # full project spec, reading list, sprint plan
```

---

## Papers

- Leviathan et al. 2023 — [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Chen et al. 2023 — [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
