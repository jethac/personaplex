# Listening matrix v2 — multi-seed, natural American female voice (NATF1)

## Design rationale
The v1 set (20260723-audio/, NATM1 voice, one seed) showed per-clip quirks
in user listening: w8a16 had rising question-like intonation on a
sentence-final word; fp8 rendered "questions" as "question"; fp8-depq8 and
w8a16-nvfp4ffn sounded good (both with an extra spoken segment). With
sampled-token divergence at frames 2-3 (see ../20260723-divergence/
DIVERGENCE.md), each clip is a DIFFERENT valid trajectory of the model —
single-clip quirks cannot be attributed to a scheme.

This v2 set therefore generates **3 seeds (42424, 1001, 2002) per scheme**
over the same 30 s deterministic input and prompt. Interpretation rule:
an artifact that recurs across all 3 seeds of one scheme (and not in the
bf16 seeds) is a real scheme artifact; one that appears in a single clip
is sampling luck. The objective quality evidence remains the
teacher-forced drift ladder (flat drift, all schemes) — this set is the
perceptual layer on top.

## Voice choice
User requested a natural American female voice. Available natural voices
in nvidia/personaplex-7b-v1 voices.tgz: NATF0-3 (female), NATM0-3 (male),
plus VARF0-4/VARM0-4 (varied/accented). **NATF1.pt** chosen as the
counterpart of the repo-default NATM1; to override, regenerate with
`python bench/audio_matrix.py --voice-prompt NATF0.pt` (or NATF2/NATF3).

## Contents
`<scheme>-s<seed>.wav` + matching `.json` (sampled text tokens), schemes:
- `bf16` — unquantized reference
- `w8a16` — weight-only 8-bit (recommended default)
- `fp8-depq8` — full-FP8 representative (amarrmb path + depformer exit)
- `w8a16-nvfp4ffn` — FFN NVFP4 over w8a16 (pass-with-caveats tier)
- `fast` — the shipped `--fast` preset (w8a16+depq8+skipother+mimi-fp16)

Generation: bench/audio_matrix.py — one model load per scheme, fresh
streaming session + prompt phase per seed, identical input
(../20260723-audio/input-30s.wav), temp 0.8/0.7, topk 250/25.

## User observations
- v1 (NATM1, seed 42424): notes above; explained as sampling-trajectory
  variation pending this multi-seed set.
- v2: (to be filled in after listening)
