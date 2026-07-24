# PR5 — Benchmark harness, pre-registered protocol, GEMV micro-lab, GB10 playbook

**Status: DRAFT — not yet opened. Stacks on PR4 (uses its flags).**

## Motivation
The repro bundle behind PRs 1-4: a headless per-frame latency harness with
serve-path parity, a pre-registered measurement protocol, a quantization
quality-gate tool, a GEMV bandwidth micro-lab, and a GB10 setup playbook —
so results are reproducible and other GB10/Spark/Thor owners can submit
comparable traces instead of one-off numbers.

## Contents
- `bench/protocol.md` — pre-registered methodology (fixed seeds, cold +
  sustained runs, p50..p99.9 + max, env block, dmon capture).
- `bench/bench.py` — headless harness reusing the real loaders/LMGen/
  warmup; per-stage CUDA-synced timers; NVTX ranges (nsys-ready via
  --profile-frames); CPU pinning; --frames / --sustained-minutes; CSV +
  summary output.
- `bench/divergence.py` — quantization quality gates: sequential
  record-then-replay bf16 reference vs schemes; free-run divergence +
  stream health (RMS/silence/click/spectral) and teacher-forced drift
  ladder; bf16-vs-bf16 self-tests must read exactly zero.
- `bench/microbench/` — batch-1 GEMV bandwidth lab (cold-rotation
  methodology) + GB10 tuning table.
- `bench/AUDIT.md`, `bench/DIVERGENCE.md`, `bench/results/ABLATIONS.md`
  — the evidence artifacts for this machine.
- `docs/gb10-playbook.md` — end-to-end GB10 setup (cu130 wheels, sphn
  build, Triton ptxas symlink, gated-model note, one-command repros).

## Evidence
This harness produced every number in PRs 1-4; its self-checks (stage sums
vs wall total, bf16 self-tests, proxy-vs-real-weight agreement within
0.1 ms) are documented inline.

## Pinned environment / tier / unknowns
As PR1. Tier: tooling only (no model-code changes). sm_121, n=1 — the
playbook explicitly invites other GB10 owners to submit env.json+summary
traces to widen n.

## Attribution
Harness, protocol, divergence gates, micro-lab and playbook authored
here (jethac). The playbook's Triton ptxas workaround follows
@amarrmb's fork README and @listerheaton's GB10-thread notes
(NVIDIA/personaplex#3); setup pitfalls cross-checked against amarrmb
commits b8d7db1/67e3203.
