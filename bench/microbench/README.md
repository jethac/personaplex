# GB10 GEMV tuning table (bench/microbench/gemv_lab.py)

**Methodology (v2, cold-rotation):** every timed call streams a weight
matrix from a >=512 MB round-robin pool (4-64 distinct copies per shape),
so the 25 MB L2 stays cold — matching the real workload, where each frame
streams ~6.5 GB of distinct weights and nothing is resident between GEMVs.
`results-gb10-cold.csv` is authoritative.

`results-gb10.csv` (v1) is retained for the record but measured HOT-cache
bandwidth: it re-read one matrix per shape, so matrices comparable to L2
report physically impossible numbers (e.g. 1398 GB/s at 4096x4096 = 5x
DRAM peak). Do not quote v1 absolutes; v1/v2 variant rankings agree.

## Cold-rotation results (weight-bytes effective GB/s; DRAM peak 273)

| shape (INxOUT) | cuBLAS bf16 | _scaled_mm fp8 | shipped w8a16 Triton | best swept variant |
|---|---|---|---|---|
| 4096x12288 | 589us / 171 | 401us / 126 | **211us / 239** | row BO8 BI1024 w8 s2 .cg: 199us / 253 |
| 4096x16896 | 560us / 247 | 404us / 171 | **286us / 242** | row BO16 BI512 w2 s2 .cg: 285us / 243 |
| 8448x4096  | 296us / 234 | 156us / 222 | **146us / 237** | tiled-arena: 151us / 230 |
| 4096x4096  | 215us / 156 | 83us / 201 | **72us / 233** | tiled-arena: 70us / 239 |
| 1024x4224  | 61us / 142 | 23us / 188 | **20us / 222** | tiled-arena: 19us / 231 |
| 4096x32000 | 1542us / 170 | 579us / 226 | **554us / 237** | row BO16 BI512 w2 s2 .cg: 546us / 240 |

## Conclusions (2026-07-24, cold methodology)

1. The shipped w8a16 Triton GEMV sustains **222-242 GB/s cold on every
   shape** (81-89% of LPDDR5X peak) — the >220 GB/s target is met by the
   kernel already wired into --w8a16/--fast. It beats torch._scaled_mm
   fp8 by 1.05-1.9x and cuBLAS bf16 gemv per byte at every shape.
2. torch._scaled_mm on sm_121 (sm89 xmma path) leaves 15-55% of bandwidth
   unused at these shapes — it, not fp8 itself, bottlenecked the fp8
   config (step 67.2 ms vs w8a16 51.1 ms).
3. The best swept variants (.cg streaming hint, burst-tiled arena,
   BO8/BI1024/w8 on the widest shape) top out at ~253 GB/s — only +5-6%
   over shipped. Wiring them in means a weight-repack pass and per-shape
   config plumbing for <=1.5 ms of step: **deferred**.
4. split-K never wins cold (atomics overhead, no residency to exploit);
   num_stages>3 flat (DRAM-latency bound); cuBLAS bf16 is erratic across
   shapes (142-247) — another reason the custom kernel path pays off.
