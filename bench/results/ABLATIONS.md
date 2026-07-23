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
| fp8 + depq8 + skipother + mimi-fp16 | **61.43** | 62.37 | 62.50 | 62.51 | 0/475 |

## step_ms (lm_gen.step only)

| config | p50 | p99 | p99.9 |
|---|---|---|---|
| bf16 (stock) | 91.35 | 93.17 | 93.28 |
| bf16 + depq8 | 81.03 | 83.05 | 83.66 |
| fp8 | 67.20 | 68.37 | 68.82 |
| fp8 + depq8 | 58.35 | 59.38 | 59.51 |
| fp8 + depq8 + skipother(+mimifp16) | 58.2-58.3 | ~59.0 | ~59.2 |

(w8a16 rows pending — cells running; will be added with the divergence
soak results.)

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
