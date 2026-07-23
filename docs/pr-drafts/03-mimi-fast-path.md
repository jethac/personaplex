# PR3 — Mimi fast path: `--skip-other-mimi` (−5.4 ms) and `--mimi-fp16` (−1.9 ms)

**Status: DRAFT — not yet opened. Stacks on PR2.**

## Motivation & discarded-work citation
The serve path runs TWO mimi instances but uses one. In `server.py`
(and mirrored in `offline.py`) every `other_mimi` result is discarded:

- L123 `_ = self.other_mimi.encode(chunk)` (warmup)
- L129 `_ = self.other_mimi.decode(tokens[:, 1:9])` (warmup)
- L225 `_ = self.other_mimi.encode(chunk)` (steady loop)
- L232 `_ = self.other_mimi.decode(tokens[:, 1:9])` (steady loop)

(line numbers at upstream 3428dfd). Its streaming state is read nowhere
else, so the stream is pure discarded work: 2x encode (~3.0 ms) + 2x
decode (~2.1 ms) per 80 ms frame. Separately, mimi runs fp32 while the LM
is bf16; fp16 mimi plus its existing `torch_compile_encoder_decoder` path
(exercised via `torch_compile_lazy`) halves mimi cost. Both ideas are from
the community GB10 effort (amarrmb fork); this PR lands them as opt-in
flags with `None`-safe plumbing, plus a dtype-safety fix in
`encode_from_sphn` (cast input batches to the mimi parameter dtype —
also from the amarrmb fork).

## Diff summary (isolated)
- `moshi/moshi/offline.py`: `--skip-other-mimi`, `--mimi-fp16`;
  `other_mimi: Optional[MimiModel]` handling in warmup/decode; warmup
  chunk dtype follows mimi dtype.
- `moshi/moshi/models/lm.py`: `encode_from_sphn` casts batches to the
  model dtype (2 lines).

Defaults unchanged (both streams, fp32) — flags are opt-in.

## Before/after (protocol, 500 frames, real weights, GB10, on top of w8a16+depq8)
| config | total p50 (ms) |
|---|---|
| w8a16 + depq8 | 53.82 |
| + skip-other-mimi | ~48.9 (−5.4: encode_other 3.0 + decode_other 2.1 + sync boundaries) |
| + mimi-fp16 (compiled) | 46.47 (encode 2.94->1.78, decode 2.11->1.45) |

(The −5.4/−1.9 splits were measured in the fp8 stack: 68.71 -> 63.33 ->
61.43; identical mimi-side deltas apply.)

## Pinned environment
(as PR1; triton 3.7.1 with system ptxas is required for the mimi compile path)

## Tolerance-ladder tier
skip-other-mimi: Tier 0/invariant (removes computation whose outputs are
discarded). mimi-fp16: Tier 1 (numeric change in the codec path; covered
by the fork's health metrics — no clicks/silence anomalies vs the bf16
reference band — and paired WAVs).

## Known unknowns
sm_121, n=1. If a future feature reads the second stream (e.g. echo
monitoring), the flag must stay off — hence opt-in. fp16 mimi on sm80/90
untested here.
