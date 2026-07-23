# Profile split of the LM step — GB10, bf16, real personaplex weights

Trace: `bench/results/20260723-profile/personaplex-bf16-50f.nsys-rep`
(50 steady-state frames via `--profile-frames 50`, nsys 2025.3.2,
`-t cuda,nvtx,osrt --cuda-graph-trace=node`, real nvidia/personaplex-7b-v1
weights, cores 5-9,15-19). A companion trace on the moshiko proxy checkpoint
(`bf16-50f.nsys-rep`) shows an identical kernel stream, confirming proxy runs
are representative. Numbers below are per frame unless noted. Under the
profiler a frame is ~105.9ms wall vs ~101.6ms clean; ratios, not absolutes,
carry the profiler overhead.

## Headline: the step is GPU-busy, not dispatch-bound

From the sqlite export (kernel intervals intersected with NVTX ranges):

| region                      | wall span | GPU busy | GPU idle gap |
|-----------------------------|-----------|----------|--------------|
| `step_ms` (lm_gen.step)     | 94.1 ms   | 91.7 ms  | **2.3 ms**   |
| whole frame                 | 105.9 ms  | 101.1 ms | 4.8 ms       |

- `process_transformer_output` (depformer graph + text/audio sampling +
  cache writes) projects to **21.6 ms** of GPU time per frame
  (nvtx_gpu_proj_sum; its CPU span is only 7.8 ms — the depformer graph
  replay completes asynchronously after the CPU exits the range).
- By subtraction, the temporal transformer forward (`forward_codes` CUDA
  graph incl. prepare/embed glue) accounts for **~70 ms** of GPU time.
- Mimi (both streams): encode 3.2 ms x2, decode 2.5 ms x2 — ~11.4 ms,
  fully eager (~654 `cudaLaunchKernel` calls per frame).

Launch/dispatch accounting per frame: **6 `cudaGraphLaunch`** + **~672
eager `cudaLaunchKernel`** (mimi + glue + text sampling), while ~9,700
kernels *execute* per frame — i.e. the two LMGen CUDA graphs already
amortize thousands of node launches; eager launch overhead lives almost
entirely in mimi and the step glue.

## Where the 91.7 ms of busy time goes

Top kernels (cuda_gpu_kern_sum, share of total GPU time):

| kernel | insts/frame | ms/frame | what it is |
|---|---|---|---|
| cuBLAS `gemvx::kernel` (bf16, 2 variants) | ~449 | **73.6** | every LM linear at batch=1 (temporal + depformer) |
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80` | 32 | 6.65 | temporal attention SDPA (context 3000) |
| `precomputed_convolve_sgemm<float>` + cudnn | ~18 | ~3.3 | mimi SEANet convs (fp32) |
| `cutlass_80_wmma_...bf16_32x32` GEMM | 96 | 2.84 | depformer per-step linears |
| `gemmSN_TN_kernel<float>` | 128 | 2.10 | mimi transformer linears (fp32) |
| `fmha_cutlassF_f32_...sm80` | 32 | 0.86 | mimi attention (fp32) |
| `fmha_cutlassF_bf16_64x64_sm80` | 96 | 0.47 | depformer attention (16 steps x 6 layers) |
| topk gather + radix sort (sampling) | 17+17 | 0.38 | audio (16) + text (1) sampling |
| ~5,000 tiny elementwise/reduce kernels | ~5000 | ~8-9 | RMSNorm decomposition, rope, copies, cache ops (~1µs each) |

**Interpretation.** ~80% of the step is cuBLAS GEMV streaming weights at
batch size 1. The LM weights are ~14-15 GB bf16 (temporal ~10.9 GB +
depformer + embeddings); 73.6 ms for one full sweep implies **~170-190 GB/s
effective bandwidth vs ~273 GB/s LPDDR5X peak** on GB10 — the gemvx kernels
leave 30-40% of bandwidth on the table, and bf16 weights are 2x the bytes
FP8 would move.

## T4 megagraph go/no-go input

Recoverable dispatch overhead:

- In-step GPU idle: **2.3 ms/frame** (gaps between the two graph replays
  and eager glue inside `lm_gen.step`).
- Frame-level stage-boundary idle: additional ~2.4 ms, partly an artifact
  of the per-stage `cudaDeviceSynchronize` timers (14/frame) and profiler
  overhead; the clean-run stage sums differ from wall total by <0.4 ms.
- Mimi eager launch overhead: ~654 launches x ~3.2 µs CPU = ~2.1 ms CPU
  time, largely overlapped; graphing mimi is worth ~1-2 ms at best.

**Estimate: a whole-frame megagraph recovers ~2-4 ms/frame (≈3%), not the
~22 ms needed to reach the 80 ms budget.** The dominant costs are (1) GEMV
bandwidth — attack with FP8 weights (measured separately, see
bench/results/ABLATIONS.md) and/or fused/better GEMV kernels, (2) the 16
sequential depformer steps (~21.6 ms; `--dep-q-exit 8` halves this), (3)
~8-9 ms of tiny-kernel busy time inside the graphs that kernel fusion
(torch.compile on norms/rope/gating elementwise) could compress, (4)
sm80-generation attention/GEMM kernel selection (see bench/AUDIT.md).

Verdict input: **megagraph alone: no-go as a primary lever; keep as a
polish item once per-kernel time dominates are addressed.**
