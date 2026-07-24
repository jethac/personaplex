# T1 kernel-selection audit — GB10 (sm_121), torch 2.13.0+cu130

Source: nsys trace `bench/results/20260723-profile/personaplex-bf16-50f.nsys-rep`
(50 steady-state bf16 frames, real personaplex weights, node-level CUDA
graph trace). "insts/frame" from cuda_gpu_kern_sum over 50 frames.

| hot op | dispatched kernel | insts/frame | ms/frame | verdict |
|---|---|---|---|---|
| LM linears, batch=1 (temporal in_proj/out_proj/gating, depformer multi-linear, lm heads) | cuBLAS `internal::gemvx::kernel<... __nv_bfloat16 ...>` (2 template variants) | ~449 | 73.6 | **cuBLAS GEMV — functional but only ~170-190 GB/s effective vs 273 GB/s peak; prime suspect.** No Blackwell-specific GEMV in the stream; not tensor-core bound (GEMV is BW-bound) but the 30-40% BW gap is real money at this size. |
| Temporal attention SDPA (context 3000) | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80` | 32 | 6.65 | **Generic sm80 memory-efficient-attention fallback.** PyTorch has no FA2/FA3/cuDNN-attention path for sm_121 here; 208µs per layer-call for q_len=1 decode-style attention is high — a decode-optimized kernel (or cuDNN attention if it gains sm_121) should cut this several-fold. |
| Depformer attention | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | 96 | 0.47 | sm80 fallback but tiny (context 8); fine. |
| Depformer per-step GEMMs | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn` | 96 | 2.84 | **sm80 WMMA tensor-op kernel on sm_121** — pre-Hopper codepath; works, leaves tensor-core perf unused, but shapes are small so ceiling is modest. |
| Mimi transformer linears (fp32) | `gemmSN_TN_kernel<float>` | 128 | 2.10 | cuBLAS small-N fp32 path. Mimi runs fp32 by design upstream; fp16/bf16 mimi (amarrmb does fp16+compile) would halve this. |
| Mimi attention (fp32) | `fmha_cutlassF_f32_...sm80` | 32 | 0.86 | sm80 fp32 fallback; small. |
| Mimi SEANet convs (fp32) | `precomputed_convolve_sgemm<float>`, `cudnn::dgrad_engine<float>` (ConvTranspose), nchw<->nhwc transposes | ~18 | ~3.3 | cuDNN/CUDA generic conv paths; the layout-transpose kernels around cudnn calls are pure overhead. Candidate for channels-last or compile. |
| RMSNorm (`norm=rms_norm_f32`) | decomposed eager chain: `mean` reduce + `pow` + `rsqrt` + 2x `mul` + casts (~5 kernels per norm, ~257 norm sites/frame) | ~1300 | ~3-4 | **Not fused** — torch decomposition, all ~1µs kernels inside the CUDA graphs. Fusion (compile or hand kernel) compresses this. |
| RoPE / cache plumbing / masking glue | `arange`, `remainder`, `where`, `index_copy`, `compare`, `bitwise_and`, `direct_copy` (~1µs each) | ~2500 | ~4-5 | Eagerly decomposed graph nodes; fusion candidate, and several (`arange` per call) are recomputed constants. |
| Sampling (audio 16 + text 1) | `sbtopk::gatherTopK` + `radixSortKVInPlace` + softmax/div | 34 | 0.38 | Generic but cheap. topk=250/25 over 2048/32000 — fine. |
| Triton kernels | none observed | 0 | 0 | Nothing routes through Triton in this stack today (ptxas swap only matters if we add Triton kernels). |

## Flags / follow-ups

1. **gemvx underutilization (biggest single item):** 73.6 ms to stream
   ~13 GB of bf16 weights. FP8 halves bytes moved (WS0); independently, a
   fused/persistent GEMV (or cuBLASLt with better heuristics for these
   shapes) targets the 30-40% BW shortfall.
2. **sm80-generation attention/GEMM kernels throughout** — nothing in the
   stream is sm_90+ or sm_121-native. Torch 2.13+cu130 ships Blackwell
   support for large GEMMs, but these decode shapes fall back to legacy
   templates. Worth re-checking after torch upgrades; not the dominant cost
   today (attention total ~8 ms/frame).
3. **~8-9 ms/frame of ~1µs elementwise kernels** (norm/rope/glue) — pure
   fusion headroom inside the existing CUDA graphs.
4. Mimi fp32 everywhere (~11.4 ms for two streams incl. both directions)
   — fp16 mimi + compile (amarrmb recipe) and/or dropping the redundant
   `other_mimi` work in the server is several ms, orthogonal to the LM.
