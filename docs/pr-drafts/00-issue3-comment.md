# DRAFT comment for NVIDIA/personaplex issue #3 (GB10/Spark real-time performance)
# Status: DRAFT — do not post until the repo owner signs off.

Following up with measured results on a GB10 machine (Lenovo ThinkStation
PGX, sm_121, aarch64, torch 2.13.0+cu130, driver 595.71.05).

**Headline: stock 101.7 ms/frame -> 48.6 ms p50 / 50.4 ms p99.9 with an
opt-in flag stack — comfortably inside the 80 ms real-time budget (0
misses in 475-frame protocol runs), with unchanged bf16 activations.**

| config | total ms p50 / p99 / p99.9 | budget misses |
|---|---|---|
| stock bf16 | 101.66 / 103.52 / 103.59 | 475/475 |
| fp8 (amarrmb's approach) | 77.61 / 78.86 / 79.50 | 0 |
| w8a16 weight-only (ours) | 61.42 / 62.78 / 63.02 | 0 |
| `--fast` (w8a16 + depformer-exit-8 + skip-other-mimi + fused norms) | **48.56 / 49.58 / 50.40** | 0 |

Key findings for this hardware class:
- The LM step is weight-streaming-bound (~449 batch-1 GEMVs, ~13 GB/frame).
- `torch._scaled_mm` (fp8) lands on an sm89 path on sm_121 and leaves
  15-55% of bandwidth unused; a weight-only 8-bit Triton GEMV (bf16
  activations) sustains 222-242 GB/s cold and beats full fp8 by 16 ms/frame
  while perturbing logits ~25-30% LESS.
- The depformer's codebooks 9..16 are always overwritten by provided user
  tokens in the serve flow — skipping them is output-invariant (−10 ms).
- The second mimi stream's outputs are discarded (server.py L123/129/225/232)
  — skipping it is free (−5.4 ms).
- Quality gates: teacher-forced drift over 1500 frames is flat (no error
  accumulation) for both fp8 and w8a16; free-running stream health matches
  the bf16 reference band; paired WAVs included.

Everything is reproducible from our fork branch:
https://github.com/jethac/personaplex/tree/pgx-plan
(bench/protocol.md = pre-registered methodology; bench/results/ =
per-frame CSVs, env blocks, ablations, divergence verdicts, audio).

One-command repro (after the playbook setup in docs/gb10-playbook.md):
```
taskset -c 5-9,15-19 python bench/bench.py --fast --frames 500 --out bench/results/repro
```

We have this staged as two PRs on our fork: **pr-build-fixes**
(torch-pin relax, meta-tensor load fix, a dtype-cast bugfix — the small
set that makes GB10 installable) and **pr-gb10-realtime** (the
performance series: amarrmb's fp8 groundwork cherry-picked with
authorship preserved, depformer early exit, mimi fast-path flags, and
the w8a16 weight-only scheme + `--fast` preset), with the bench
harness/playbook available as a follow-up PR on request. Happy to open
them if maintainers are interested. Credit to @amarrmb for the fp8
groundwork that started this line of work, and to @gplv2 / @acatovic /
@listerheaton for the thread findings we built on.
