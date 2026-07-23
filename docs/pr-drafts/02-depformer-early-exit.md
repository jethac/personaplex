# PR2 — Depformer early exit (`--dep-q-exit 8`): −10 ms/frame, output-invariant

**Status: DRAFT — not yet opened. Stacks on PR1 (GB10 installability).**

## Motivation
`get_moshi_lm` expands dep_q to 16, so `LMGen.depformer_step` runs 16
sequential depth-transformer steps per frame. The serve stack decodes agent
audio from codebooks 1..8 only; codebooks 9..16 model the user side.

## The invariance argument
In every step of the serve flow the user-side codebooks are **provided**:
- streaming: `lm_gen.step(codes)` feeds real user audio codes each frame;
- warmup: encoded zero-frames are fed identically;
- voice/text prompt phases: `_step_voice_prompt_frame`,
  `_step_audio_silence_core`, `_step_text_prompt_core` all pass
  `input_tokens=self._encode_sine_frame()`.

`process_transformer_output` writes sampled tokens into the cache with
`torch.where(~provided, sampled, cache)` — for always-provided channels the
sampled depformer outputs for codebooks 9..16 are **always discarded**.
Skipping their computation cannot change any output the serve path uses.
Guards enforce the precondition: `step()` raises if `input_tokens` is
missing while early exit is active; the flag rejects
`return_logits`/`report_loss` and N<8.

## Diff summary (isolated)
- `moshi/moshi/models/lm.py`: `LMGen(depformer_early_exit=None)` statically
  shortens the CUDA-graph-captured depformer loop and pads skipped
  codebooks with `zero_token_id` (+33 lines incl. guards/comments).
- `moshi/moshi/offline.py`: `--dep-q-exit` CLI flag.

## Before/after (pre-registered protocol: 500 frames, real weights, GB10)
| config | total p50/p99/p99.9 (ms) | step p50 |
|---|---|---|
| bf16 | 101.66 / 103.52 / 103.59 | 91.35 |
| bf16 + depq8 | 91.43 / 93.59 / 94.49 | 81.03 |
| w8a16 (PR4) | 61.42 / 62.78 / 63.02 | 51.06 |
| w8a16 + depq8 | 53.82 / 54.91 / 55.34 | 43.59 |

Effect: −10.3 ms (bf16) / −7.5 ms (w8a16) per frame, additive with other
optimizations. Weight-traffic arithmetic: dep_q 8..15 slices are ~half of
the depformer's 1.36B params streamed per frame.

## Pinned environment
(as PR1; full env.json in the fork's bench/results/ dirs)

## Tolerance-ladder tier
Tier 0/invariant — bit-identical serve-path outputs by construction (the
skipped values are overwritten by provided tokens before any use).

## Known unknowns
sm_121, n=1. Batch>1 serving and `report_loss` training flows untested
with the flag (guarded off). Anyone feeding step() without user codes gets
a hard error by design.
