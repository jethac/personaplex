# PR4 — w8a16 weight-only quantization + Triton GEMV + `--fast` preset (2.2x)

**Status: DRAFT — not yet opened. Stacks on PR3. Headline PR of the series.**

## Motivation
On GB10 the LM step is memory-bandwidth-bound: ~449 batch-1 cuBLAS bf16
GEMVs stream ~13 GB of weights per 80 ms frame at only ~170-190 GB/s
effective (nsys, bench/AUDIT.md in the fork). Stock serve latency is
101.7 ms/frame — the model cannot meet its own 80 ms real-time budget on
this hardware.

## What this adds
- `moshi/moshi/w8a16_quantize.py`: weight-only 8-bit quantization —
  weights stored `float8_e4m3fn` with per-output-channel scales,
  dequantized in-register by a hand-written Triton GEMV (fp32
  accumulate); **activations stay bf16** (no activation scaling, no fp8
  matmul). Cold-measured 222-242 GB/s on every LM shape (81-89% of
  LPDDR5X peak; methodology + sweep in the fork's bench/microbench/).
- `moshi/moshi/fp8_quantize.py`: amarrmb's full-FP8 path
  (github.com/amarrmb/personaplex), included with credit as the
  alternative and comparison baseline — it pioneered quantizing this
  model on GB10.
- `offline.py`: `--w8a16`, `--fp8`, and `--fast` (= `--w8a16
  --dep-q-exit 8 --skip-other-mimi --mimi-fp16`).

## Why weight-only beats full FP8 here
`torch._scaled_mm` dispatches to an sm89 path on sm_121 reaching only
126-226 GB/s at these shapes; the Triton GEMV sustains 222-242 GB/s.
End-to-end (protocol, 500 frames, real weights): step 91.4 (bf16) ->
67.2 (fp8) -> **51.1 ms (w8a16)**. w8a16 also shows ~25-30% smaller
per-frame logit deltas than fp8 (weight-only, activations untouched).

## Before/after distributions (pre-registered protocol, GB10)
| config | total p50 / p99 / p99.9 (ms) | budget misses |
|---|---|---|
| bf16 stock | 101.66 / 103.52 / 103.59 | 475/475 |
| fp8 | 77.61 / 78.86 / 79.50 | 0/475 |
| w8a16 | 61.42 / 62.78 / 63.02 | 0/475 |
| **--fast** | **46.47 / 47.58 / 48.11** | 0/475 |

2.2x vs stock; ~34 ms of tail headroom under the 80 ms/frame budget.

## Quality evidence (tolerance-ladder tier)
Tier 1 — PASS (fork's bench/results/20260723-divergence/DIVERGENCE.md):
teacher-forced drift over 1500 frames is FLAT (log-log exponents ~0,
decisively sublinear; harness self-test measures exactly 0.0 on
bf16-vs-bf16); free-running stream health (silence/click/spectral) sits
inside the bf16 reference band on the same seeds; paired 30 s WAVs
(same seed/input/prompt) provided for listening.

## Pinned environment
(as PR1; Triton 3.7.1 functional on sm_121 requires the system-ptxas
symlink over triton's bundled ptxas-blackwell.)

## Known unknowns
sm_121, n=1. Kernel block configs tuned on GB10 only (two shape-class
configs; may need retune elsewhere). Task-level metrics (WER etc.) not
measured — numeric drift and signal health only. Uncalibrated per-channel
absmax scaling.
