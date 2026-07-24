# Ablations — per-frame latency, GB10 (ThinkStation PGX)

Protocol: `bench/protocol.md` (500 frames, seed 42424, per-stage CUDA-synced
timers, cores 5-9,15-19, warm_skip 25 -> 475 frames summarized). Weights:
**real `nvidia/personaplex-7b-v1`**, voice prompt `NATM1.pt`, default
sampling. Env: torch 2.13.0+cu130, driver 595.71.05, GB10 sm_121, GPU
otherwise idle. Budget: one 80 ms frame per 80 ms.

Rows above the line keep full serve-path parity (both mimi streams, fp32
mimi). Rows below also change the mimi path (still headless serve-shape,
but stage mix differs — see flags).

## total_ms (full frame)

| config | p50 | p99 | p99.9 | max | 80ms misses |
|---|---|---|---|---|---|
| bf16 (stock) | 101.66 | 103.52 | 103.59 | 103.61 | 475/475 |
| bf16 + depq8 | 91.43 | 93.59 | 94.49 | 94.64 | 475/475 |
| fp8 | 77.61 | 78.86 | 79.50 | 79.70 | **0/475** |
| fp8 + depq8 | 68.71 | 69.75 | 70.28 | 70.56 | **0/475** |
| --- | | | | | |
| fp8 + depq8 + skip-other-mimi | 63.33 | 64.50 | 64.79 | 64.81 | 0/475 |
| fp8 + depq8 + skipother + mimi-fp16 | 61.43 | 62.37 | 62.50 | 62.51 | 0/475 |
| w8a16 | 61.42 | 62.78 | 63.02 | 63.10 | 0/475 |
| w8a16 + depq8 | 53.82 | 54.91 | 55.34 | 55.44 | 0/475 |
| w8a16 + depq8 + rmsnorm-fusion | 53.31 | 54.46 | 54.99 | 55.30 | 0/475 |
| --fast (w8a16+depq8+skipother+mimifp16+fusion) | 46.47 | 47.58 | 48.11 | 48.11 | 0/475 |
| **--fast --nvfp4-ffn** | **41.78** | **43.16** | **43.79** | 44.05 | 0/475 |

## step_ms (lm_gen.step only)

| config | p50 | p99 | p99.9 |
|---|---|---|---|
| bf16 (stock) | 91.35 | 93.17 | 93.28 |
| bf16 + depq8 | 81.03 | 83.05 | 83.66 |
| fp8 | 67.20 | 68.37 | 68.82 |
| fp8 + depq8 | 58.35 | 59.38 | 59.51 |
| fp8 + depq8 + skipother(+mimifp16) | 58.2-58.3 | ~59.0 | ~59.2 |
| w8a16 | 51.06 | 52.40 | 52.65 |
| w8a16 + depq8 | 43.59 | 44.62 | 44.71 |
| w8a16 + depq8 + rmsnorm-fusion | 42.95 | 44.07 | 44.55 |
| --fast | 43.16 | 44.26 | 44.70 |
| --fast --nvfp4-ffn | 38.39 | 39.70 | 40.15 |

--fast composition: --w8a16 --dep-q-exit 8 --skip-other-mimi --mimi-fp16
plus the rms_norm torch_compile_lazy fusion (in-tree). w8a16 is the
weight-only 8-bit Triton GEMV path (bf16 activations); it BEATS full fp8
by ~16 ms of step because torch._scaled_mm dispatches to a slow sm89
path on sm_121 while the Triton kernel sustains 232-242 GB/s — see
bench/microbench/README.md. rms_norm fusion is worth ~0.5 ms of step. NVFP4 on the temporal FFN
(--nvfp4-ffn, packed e2m1 + e4m3 block scales, Triton dequant GEMV) takes
another ~4.8 ms of step; quality standing in
bench/results/20260723-divergence/DIVERGENCE.md (pass-with-caveats,
uncalibrated). Full ladder: stock bf16 101.66 -> 41.78 ms p50 (2.43x),
every optimized cell 0/475 budget misses.

## Attribution / notes

- **fp8 alone already clears the 80 ms budget** (max 79.70); fp8+depq8
  gives ~11 ms tail headroom; the mimi-side adds bring p50 to **61.4 ms**
  (~19 ms headroom). Neither bf16 cell ever makes budget.
- Effects compose additively: fp8 -24.2 ms step, depq8 -10.3 ms step,
  skip-other-mimi -5.4 ms total (encode_other 3.0 + decode_other 2.1 +
  stage-boundary syncs), mimi-fp16+compile -1.9 ms total (encode 2.94 ->
  1.78, decode 2.11 -> 1.45; torch.compile/Triton works on sm_121 with the
  system-ptxas symlink).
- **skip-other-mimi safety**: every `other_mimi.encode/decode` result in
  server.py is assigned to `_` and discarded (lines 123/129/225/232 at
  3428dfd) and its state feeds nothing — the stream is pure discarded
  work. Kept ON by default in the protocol rows above the line for parity
  with stock server.py.
- **Attention (T1) measured no-go**: for the decode shape (q=1, kv=3000,
  H=32, d=128, bf16) FLASH (no-mask), CUDNN, and EFFICIENT backends all
  land at 200-222 us/call (6.5-7.1 ms/frame x32); the op is KV-cache
  bandwidth-bound (~1.57 GB/frame -> ~5.7 ms floor at 273 GB/s), so no
  backend swap or custom kernel recovers meaningful time. The sm80-named
  fmha kernel is already ~87% of the bandwidth bound. MATH backend: 46 ms
  (avoid).
- amarrmb reference: their 74.2 ms total included skip-other + fp16 mimi;
  our equivalent config measures 61.4 ms (and 63.3 with fp32 mimi).
- Quality: depq8 is provably output-invariant in the serve flow; fp8 is
  not — see the divergence soak (DIVERGENCE.md, pending) and paired WAVs
  in bench/results/20260723-audio/ (bf16 / fp8 / fp8-depq8 / w8a16, same
  seed, prompt, and 30 s deterministic input).
