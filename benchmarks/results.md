# Benchmark Results

*(populated during Week 2)*

## Hardware

- Machine: Apple M2 Max MacBook
- Backend: MPS / CPU
- PyTorch version:
- Models: distilgpt2 (draft), gpt2-medium (verifier)

## Throughput

| Method | K | Temp | α | tok/s | Speedup |
|---|---|---|---|---|---|
| Greedy (verifier) | — | 0.0 | — | | 1.0x |
| Greedy (draft only) | — | 0.0 | — | | |
| Speculative | 2 | 0.0 | | | |
| Speculative | 4 | 0.0 | | | |
| Speculative | 8 | 0.0 | | | |
| Speculative | 4 | 0.5 | | | |
| Speculative | 4 | 1.0 | | | |

## Acceptance Rate by Temperature

| K | Temp=0.0 | Temp=0.5 | Temp=1.0 | Temp=1.5 |
|---|---|---|---|---|
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |

## Sequence Length Sweep (K=4, Temp=0.0)

| Prompt length | α | tok/s | Speedup |
|---|---|---|---|
| 32 | | | |
| 128 | | | |
| 512 | | | |
