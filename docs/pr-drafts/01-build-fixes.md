# PR1 — GB10/Blackwell installability: relax torch pin, fix meta-tensor load crash

**Status: DRAFT — not yet opened. For review by repo owner before submission.**

## Motivation
PersonaPlex cannot currently be installed or loaded on NVIDIA GB10-class
machines (DGX Spark, ThinkStation PGX, Jetson Thor — sm_121, aarch64):

1. `moshi/pyproject.toml` pins `torch < 2.5`, but sm_121 requires the CUDA
   13.0 aarch64 wheels (torch >= 2.9, `download.pytorch.org/whl/cu130`).
   Community GB10 threads carry this fix already (amarrmb fork does too).
2. `loaders.get_moshi_lm` initializes the LM on the meta device (PR #18
   init-OOM fix). Any checkpoint key that is neither present nor covered by
   the dep_q 8->16 backfill patches survives
   `load_state_dict(strict=False, assign=True)` as a meta tensor and
   crashes the final `.to()` with "Cannot copy out of meta tensor". This
   fires for base-Moshi dep_q=8 checkpoints (`depformer_emb.7.weight` is
   not in the checkpoint and not covered by the 8..15 backfill).

## Diff summary (isolated)
- `moshi/pyproject.toml`: `torch >= 2.2.0, < 2.5` -> `torch >= 2.2.0` (1 line).
- `moshi/moshi/models/loaders.py`: after the dep_q backfill, zero-init any
  model key still absent from the checkpoint, with a warning (13 lines).
  No behavior change for personaplex-7b-v1 checkpoints (all keys present).

## Evidence
Verified on the env below: stock repo fails to pip-install (pin) and, with
the pin fixed, fails to load kyutai/moshiko-pytorch-bf16 (meta crash);
with this PR both work, and personaplex-7b-v1 loads byte-identically
(loader path unchanged when no keys are missing).

## Pinned environment
- Lenovo ThinkStation PGX, NVIDIA GB10 (sm_121, 48 SMs, 25MB L2), aarch64,
  128GB unified LPDDR5X (~273 GB/s), 20-core Grace (Cortex-X925/A725)
- Ubuntu 24.04, driver 595.71.05, CUDA 13.0/13.2, python 3.12.3
- torch 2.13.0+cu130, triton 3.7.1 (bundled ptxas-blackwell replaced by
  system /usr/local/cuda/bin/ptxas 13.0.88 via symlink)

## Tolerance-ladder tier
Tier 0 (build/load correctness; no numerics).

## Known unknowns
sm_121, n=1 machine. The relaxed pin is unbounded above; maintainers may
prefer `< 3.0`. Not tested on sm_80/sm_90 (no hardware); the loader change
is inert when checkpoints are complete.

## Attribution
- torch-pin relax: first shipped in @amarrmb's fork (commit 94cbbbd,
  github.com/amarrmb/personaplex); the pin conflict was reported in the
  GB10 discussion thread (NVIDIA/personaplex#3, @acatovic). This PR
  carries the same one-line change with a Co-authored-by trailer.
- meta-tensor zero-init fix: authored here (jethac), interacting with
  upstream PR #18's meta-device init.
- @amarrmb: please flag any attribution adjustment you'd like — happy to
  amend.


## Update (consolidated series)
This branch (`pr-build-fixes`) now also carries @amarrmb's
encode_from_sphn dtype-cast bugfix, cherry-picked with authorship
preserved (their add7726) - it is a correctness fix independent of the
performance series. The performance work is consolidated into a single
second PR (`pr-gb10-realtime`, see 02-gb10-realtime.md), which is based
on this branch.

## Upstream overlap (pre-flight scan, 2026-07-24)
- **PR #78** (@SuperMarioYL, Apr 8, open/unreviewed) widens ALL dependency
  version ranges in pyproject.toml and overlaps our torch-pin change.
  Credit to them for flagging the install failures broadly. Ours is
  deliberately narrower: a torch-only relax, since blanket-unpinning every
  dependency risks breakage from untested majors. We would happily rebase
  this PR on #78 if the maintainer prefers the broader approach.
- **PR #63** (@haosenwang1018, Feb 23) removes the same torch upper bound
  (motivated by Python 3.13 support) — same one-line intent as our pin
  relax; credit to them as an earlier report of the constraint. Either PR
  merging first makes that hunk of ours a no-op rebase.
- File-level only: #100/#76/#37/#32/#61 also touch pyproject.toml for
  unrelated reasons (macOS/Nix, client fixes, UI, stdio runtime, Docker).
