# M1 ablations — per-frame latency, GB10 (ThinkStation PGX)

Protocol: `bench/protocol.md` (500 frames, seed 42424, serve-path parity:
2x mimi encode + lm_gen.step + 2x mimi decode + D2H, cores 5-9,15-19,
per-stage CUDA-synced timers, warm_skip 25 -> 475 frames summarized).
Weights: **real `nvidia/personaplex-7b-v1`** (HF access granted 2026-07-23),
voice prompt `NATM1.pt`, default sampling (temp 0.8/0.7, topk 250/25).
Env: torch 2.13.0+cu130, CUDA driver 595.71.05, GB10 sm_121, GPU otherwise
idle. Full per-cell env blocks and time-series CSVs in the sibling dirs.

Budget: one 80 ms frame per 80 ms.

## total_ms (full serve-path frame)

| config       | p50    | p99    | p99.9  | max    | 80ms misses |
|--------------|--------|--------|--------|--------|-------------|
| bf16 (stock) | 101.66 | 103.52 | 103.59 | 103.61 | 475/475     |
| bf16 + depq8 |  91.43 |  93.59 |  94.49 |  94.64 | 475/475     |
| fp8          |  77.61 |  78.86 |  79.50 |  79.70 | **0/475**   |
| fp8 + depq8  |  68.71 |  69.75 |  70.28 |  70.56 | **0/475**   |

## step_ms (lm_gen.step only)

| config       | p50   | p99   | p99.9 |
|--------------|-------|-------|-------|
| bf16 (stock) | 91.35 | 93.17 | 93.28 |
| bf16 + depq8 | 81.03 | 83.05 | 83.66 |
| fp8          | 67.20 | 68.37 | 68.82 |
| fp8 + depq8  | 58.35 | 59.38 | 59.51 |

## Notes

- **fp8 alone already clears the 80 ms budget** (max 79.70 over 475
  frames — thin margin); **fp8 + depq8 gives ~11 ms of tail headroom**
  (max 70.56). Neither bf16 cell ever makes budget.
- Effects compose almost exactly: fp8 -24.2 ms step, depq8 -10.3 ms
  (bf16) / -8.9 ms (fp8), combined -33.0 ms.
- amarrmb reference on DGX Spark: lm_step ~70 ms with fp8; we measure
  67.2 ms with their quantization recipe (and their 74.2 ms total also
  skipped the second mimi stream, which this protocol keeps for parity).
- torch 2.13.0+cu130 note: the fp8 path (`torch._scaled_mm` with
  per-tensor scales) ran **unmodified** — no API drift from the
  torch 2.9/2.10 the amarrmb fork targeted. fp8_quantize.py reports 321
  quantized Linears + 32 in_proj weights; quantize takes 0.33 s at load.
- Proxy validation: the pre-access baseline on kyutai/moshiko-pytorch-bf16
  (architecture-identical) measured total p50 101.58 / step p50 91.28 vs
  101.66 / 91.35 on real weights — proxy latency numbers transfer within
  ~0.1 ms.
- Quality/artifact metrics for fp8 and depq8 are NOT covered here
  (latency-only protocol); depq8 is provably output-invariant in the
  serve flow (skipped codebooks are always overwritten by provided user
  tokens), fp8 is not (weight quantization) and needs a listening/metric
  pass in a later milestone.
- Cold start (fp8 cell): mimi 5.7 s, LM load 135.9 s, quantize 0.3 s,
  warmup 2.3 s, voice+text prompt phase 6.2 s -> 150.7 s total.
