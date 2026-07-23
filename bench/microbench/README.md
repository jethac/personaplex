# GB10 GEMV tuning table (bench/microbench/gemv_lab.py)

Full sweep: `results-gb10.csv`. Effective GB/s counts weight bytes only.
CAVEAT: the bench loop re-reads the same weight matrix, so matrices
comparable to the 25 MB L2 get partial cache residency and read above-DRAM
numbers (e.g. 4096x4096 fp8 = 16 MB shows >1 TB/s). Cross-variant
comparisons within a shape remain valid; in-model absolute numbers are
lower (nsys-measured cuBLAS bf16 in the full model: ~170-190 GB/s).

| shape (INxOUT) | cuBLAS bf16 | _scaled_mm fp8 | shipped w8a16 triton | best variant |
|---|---|---|---|---|
| 4096x12288 | 597us / 168 GB/s | 350us / 144 | **213us / 236** | tiled-arena BO16 BI512 w2 s3: 199us / 253 |
| 4096x16896 | 584us / 237 | 402us / 172 | **292us / 237** | row BO16 BI512 w2 s2 .cg: 282us / 245 |
| 8448x4096  | 276us / 251 | 158us / 219 | **149us / 232** | tiled-arena: 152us / 228 |
| 4096x4096  | 183us / 183 | 68us / 247 | **16us / 1073*** | splitk2: 12us / 1398* (*L2-resident) |
| 1024x4224  | 9.6us / 897* | 13us / 332* | 13us / 333* | tiled-arena: 7.2us / 604* |
| 4096x32000 | 1506us / 174 | 599us / 219 | **542us / 242** | row BO16 BI512 w2 s2: 541us / 243 |

Conclusions (2026-07-24):

1. The shipped w8a16 Triton GEMV (BO16/BI512 for IN>=4096, BO16/BI256/w4
   otherwise) already reaches the >220 GB/s target on every large shape and
   beats torch._scaled_mm fp8 by 1.2-1.9x — this is why w8a16 beats full
   fp8 end-to-end (step 51.1 ms vs 67.2 ms).
2. torch._scaled_mm (sm89 xmma path on sm_121) is the bottleneck of the
   fp8 config, not fp8 itself — matches the sglang-GB10 finding.
3. Best additional variants (burst-tiled weight arena, .cg streaming
   hint) buy only +2-7% (253 GB/s peak = 93% of LPDDR5X). Wiring the
   tiled arena in requires a weight repack pass for ~1-2 ms of step —
   deferred; the CSV documents the recipe (BO16 BI512 warps2 stages3).
4. split-K helps only L2-resident shapes (no DRAM win); cp.async staging
   beyond stages=2-3 shows no further gain (DRAM-latency bound, not
   issue-bound).
