# PR2 — GB10 real-time serving: quantization + serve-path optimizations (2.2x, 101.7 -> 46.5 ms/frame)

**Status: DRAFT — not yet opened. Depends on PR `pr-build-fixes` (GitHub
will show its commits combined until that merges).**

## Problem
PersonaPlex must produce one 80 ms audio frame every 80 ms. On NVIDIA
GB10 (DGX Spark / ThinkStation PGX / Jetson Thor class, sm_121, unified
LPDDR5X ~273 GB/s) stock bf16 measures ~101.7 ms/frame — real-time is
unreachable. Profiling (nsys, evidence links below) shows the LM step is
weight-streaming-bound: ~449 batch-1 GEMVs move ~13 GB of bf16 weights
per frame at only ~170-190 GB/s effective; dispatch overhead is <3 ms, so
the recoverable costs are weight bytes, redundant serve-path work, and
the depformer's unused half.

## What this PR adds (ordered commit story)
1. **amarrmb's FP8 groundwork** (cherry-picked with authorship
   preserved): `fp8_quantize.py` (torch._scaled_mm dynamic FP8), plus
   server-path optimizations (skip discarded second mimi stream, pinned
   DtoH, fp16+compiled mimi, frame profiling).
2. **Depformer early exit** (`--dep-q-exit 8`): dep_q=16 runs 16
   sequential depth steps but the serve flow always provides the
   user-side codebooks, so sampled outputs for codebooks 9..16 are
   always overwritten before use — skipping them is output-invariant
   (guards enforce the precondition). −10.3 ms (bf16) / −7.5 ms (w8a16).
3. **Mimi fast-path flags** (`--skip-other-mimi`, `--mimi-fp16`):
   amarrmb's server-side ideas as opt-in offline.py flags with None-safe
   plumbing (−5.4 ms and −1.9 ms).
4. **Quantization scaffolding + `--fp8` wiring** (derived work,
   Co-authored-by @amarrmb): offline-path wiring of their quantizer and
   the shared layer-selection/patching scaffolding in w8a16_quantize.py
   (module walk, min_features gate, depformer-self_attn skip, in_proj
   handling, class-patch machinery), with a naive dequant baseline.
5. **w8a16 kernel + presets** (original work): weights stored fp8-e4m3
   with per-output-channel scales, dequantized in-register by a
   hand-written Triton GEMV; **activations stay bf16**. The GEMV
   sustains 222-242 GB/s cold vs 126-226 for torch._scaled_mm on sm_121
   (which lands on an sm89 path), so weight-only beats full FP8 by
   ~16 ms/frame while perturbing logits ~25-30% less. `--fast` =
   `--w8a16 --dep-q-exit 8 --skip-other-mimi --mimi-fp16`.

**What is original here:** the Triton dequant-in-register GEMV, the
per-channel w8a16 scheme built on it, the `--fast` preset, and the
finding that weight-only 8-bit outperforms full FP8 on sm_121 (a
kernel-dispatch effect, not a numerics one) — plus the depformer
invariance proof and all measurements/quality gates. The FP8 quantizer,
the serve-path optimizations, and the quantization scaffolding are
amarrmb's work, carried with commit-level attribution.

## Ablation ladder (pre-registered protocol, 500 frames, real weights, GB10)
| config | total ms p50 / p99 / p99.9 | budget misses |
|---|---|---|
| bf16 stock | 101.66 / 103.52 / 103.59 | 475/475 |
| fp8 | 77.61 / 78.86 / 79.50 | 0 |
| w8a16 | 61.42 / 62.78 / 63.02 | 0 |
| w8a16 + depq8 | 53.82 / 54.91 / 55.34 | 0 |
| **--fast** | **46.47 / 47.58 / 48.11** | 0 |
| fast with fp32 mimi (variant) | 48.56 / 49.58 / 50.40 | 0 |

## Sustained validation (30 min, dmon alongside)
38,453 warm frames: p50 46.65 / p99 47.75 / **p99.9 48.20 / max 48.96 ms
— zero budget misses**. GPU 51->66 C, SM clocks −1.8%, 33->37 W: no
thermal cliff; the headline holds warm.

## Quality evidence (tolerance ladder)
- Teacher-forced drift (bf16 reference trajectory forced into each
  scheme, 3 seeds x 1500 frames; harness self-test = exactly 0.0):
  drift exponents ~0 for fp8 and w8a16 — **flat, no error
  accumulation; decisively sublinear = PASS.**
- Free-running stream health (5 seeds x 5000 frames): silence/click/
  spectral statistics inside the bf16 reference band; zero clicks >0.25
  in ~30k decoded frames.
- Perceptual layer: multi-seed listening matrix (5 schemes x 3 seeds,
  natural-female voice) in the fork's bench/results/20260724-audio-v2/.
- Full verdicts: bench/results/20260723-divergence/DIVERGENCE.md on the
  fork's pgx-plan branch.

## Reproduction / evidence links (fork, branch pgx-plan)
https://github.com/jethac/personaplex/tree/pgx-plan — bench/protocol.md
(pre-registered methodology), bench/bench.py + divergence.py (harness &
quality gates), bench/microbench/ (GEMV bandwidth lab + GB10 tuning
table), bench/results/ (per-frame CSVs, env blocks, ablations, soak,
audio). The harness + GB10 playbook are available as a follow-up PR on
request.

## Pinned environment
Lenovo ThinkStation PGX — GB10 (sm_121, 48 SMs, 25 MB L2), aarch64,
128 GB unified LPDDR5X; Ubuntu 24.04, driver 595.71.05, CUDA 13.0;
python 3.12.3, torch 2.13.0+cu130, triton 3.7.1 with system ptxas
symlinked over the bundled ptxas-blackwell.

## Behavioral guarantees

**Zero change without flags.** Every optimization is opt-in; each verbatim
contributor cherry-pick is immediately followed by a gating fixup commit so
that at the branch tip, stock invocation matches upstream. Evidence: a
seeded no-flag offline run (seed 42424, 30 s input, 375 frames, real
weights) on the branch tip vs upstream/main produced **byte-identical
output WAV and byte-identical token stream** (`cmp` on both artifacts).
The only remaining non-numeric deltas are (a) a per-50-frames timing log
line in the server loop (observability only; happy to gate or drop it on
request) and (b) one informational log line at startup on GB10-class
devices (sm_121) when no perf flag is active, pointing users at the
opt-in flags — added as the FINAL commit of the series so a maintainer
can drop it independently; it fires on no other arch and never when any
perf flag is set.

**Per-flag prerequisites** (checked at startup/model-build time with
actionable RuntimeErrors; no silent fallbacks — if a flag cannot meet its
performance contract, it errors rather than quietly degrading):

| flag | requirement | check |
|---|---|---|
| `--fp8` | CUDA, compute capability >= 8.9, `torch._scaled_mm` | explicit startup check in server and offline wiring |
| `--w8a16` | CUDA + Triton | module imports without Triton (lazy kernel builder, built+cached on first use); quantize errors with install hint if Triton missing |
| `--mimi-fp16` | working torch.compile backend (Triton on CUDA) | checked at argument parse time. Perceptual note: a single-trial report of onset softening was falsified by a 5-pair blind matched-pairs A/B (0/5 discrimination; onset character tracks the sampling seed, not mimi precision) — part of --fast |
| `--dep-q-exit` | none (pure logic) | precondition guard: rejects steps without user input tokens |
| `--skip-other-mimi` | none (pure logic) | n/a |
| `--pinned-io` | CUDA | n/a (allocation-time) |
| `--fast` | composes the above | composed startup check that names the specific missing prerequisite |

No architecture-sniffing anywhere: capability checks only, no `sm_121`
conditionals; the Triton ptxas symlink workaround is documentation-only
(GB10 playbook), never code. The Triton GEMV is generic sm_80+.

## Attribution
| what | who | provenance |
|---|---|---|
| fp8_quantize.py, server-path opts (skip-other, pinned DtoH, fp16 mimi) | @amarrmb | cherry-pick -x of amarrmb/personaplex 94cbbbd (also validated by them: 74 ms on DGX Spark) |
| encode_from_sphn dtype cast (in PR `pr-build-fixes`) | @amarrmb | cherry-pick -x of add7726 |
| depformer early-exit observation | @gplv2 | NVIDIA/personaplex#3 discussion |
| early-exit implementation + invariance proof + guards | jethac | this PR |
| mimi flags adaptation (offline path) | jethac, Co-authored-by @amarrmb | this PR |
| quantization scaffolding (layer walk/gates/class patches) + --fp8 wiring | derived from @amarrmb's fp8_quantize.py | this PR, commit "quantization scaffolding", Co-authored-by trailer |
| w8a16 scheme, Triton dequant GEMV, per-channel scaling, --w8a16/--fast, all measurements & quality gates | jethac | this PR, commit "w8a16: weight-only..." (no co-author: original work) |
| torch-pin conflict report / ptxas notes | @acatovic / @listerheaton | NVIDIA/personaplex#3 |

@amarrmb: please flag any attribution adjustment you'd like — happy to
amend.

## Gating notes
- `--mimi-fp16` guidance and any default-scheme wording await the
  user's listening pass over the multi-seed matrix; until then all
  flags are opt-in and defaults are unchanged.
- Known unknowns: sm_121, n=1 machine (plus amarrmb's Spark/Thor for
  the fp8/server parts); task-level metrics (WER etc.) not measured.
