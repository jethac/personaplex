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

## User observations — v2 results (recorded verbatim)
- fp8-depq8: perfect at all 3 seeds.
- w8a16 and w8a16-nvfp4ffn: both perfect at s1001/s42424; both have "a
  little strange" final intonation at s2002.
- fast: s2002 good; s1001 odd phrasing ("hey let me know if you have any
  question"); s42424 onset degraded ("hey" sounds like "eyy").
- Follow-up listen of bf16-s2002: "weird at FULL precision — a
  truncation, then it goes off to talk about something completely
  different."

## Findings
**s2002 is a hard seed; w8a16 exonerated on the s2002 count.** Per-seed
cross-scheme comparisons are different trajectories (divergence at frames
2-3); s2002 produces oddities even in bf16, so the w8a16/nvfp4ffn s2002
intonation notes fall within the reference's own weirdness band, and
fp8-depq8's perfect s2002 was trajectory luck, not a scheme property.

Objective corroboration (torchaudio pitch contour over the last voiced
2 s of each clip; corroborates, does not decide):

| clip | speech ends | final-400ms f0 slope |
|---|---|---|
| bf16-s2002 | 13.2 s (truncated) | **+51 Hz/s (rising)** |
| w8a16-nvfp4ffn-s2002 | 26.8 s | −15 Hz/s |
| fp8-depq8-s2002 | 1.8 s* | −55 Hz/s |
| bf16-s1001 / bf16-s42424 | ~1.8 s* | −72 / −108 Hz/s (falling) |
| w8a16-s1001 / w8a16-s42424 | ~1.9 s* | −38 / −6 Hz/s (falling) |

(*short clips: the agent said one short line then stayed silent.) The
full-precision reference itself carries the rising-final-contour anomaly
at s2002; normal seeds show natural falling declarative contours in
every scheme.

**Open question: the fast onset artifact at s42424/s1001** — under
isolation via fastfp32mimi-s{42424,1001}.wav (identical stack, mimi at
fp32); see the A/B verdict below/in the report.

## mimi-fp16 isolation A/B verdict (fastfp32mimi-s{42424,1001})
Setup: identical stack to --fast but mimi at fp32, same seeds/input.
Results:
- **Text tokens identical** to the fast clips at both seeds (same 11
  tokens, same sentence).
- Speech onset at 0.38 s; the first 0.5 s of PCM is numerically
  near-identical (rel L2 0.024 / 0.003 — inaudible codec noise). The
  "hey"->"eyy" attack lies inside this shared prefix, so **the onset
  artifact does not vanish at fp32: mimi-fp16 codec rendering is NOT
  the culprit** (per the pre-agreed criterion, not implicated; --fast
  composition unchanged pending user confirmation).
- From 0.5 s the audio-channel sampling diverges (rel L2 > 1 through
  the spoken sentence, back to ~0.07 codec-noise floor in silence):
  fp16-vs-fp32 mimi ENCODE perturbs the LM input codes, so the two
  runs render the same sentence as different audio trajectories — the
  same divergence class as quantization (frames 2-3), not a rendering
  defect.
- Prediction for user listen: fastfp32mimi-s42424.wav should carry the
  SAME "eyy" onset as fast-s42424.wav. If confirmed, the artifact is
  trajectory luck; if it sounds clean, the criterion above is wrong and
  we demote mimi-fp16 from --fast.

## AMENDMENT: the prediction FAILED — mimi-fp16 demoted from --fast
User listening (blind-ish A/B): fast-s42424 has the "eyy" onset;
fastfp32mimi-s42424 has a clean "hey". The numeric analysis above
(0.003 rel L2 / 0.003 spectral diff over the first 0.5 s) predicted no
audible difference and was WRONG: a perceptually salient difference in
the ~20 ms attack transient hid inside metrics dominated by the rest of
the window. Standing lesson, recorded for the protocol: transient-attack
differences evade sub-second L2/spectral aggregates; human listening is
reserved as a release gate precisely because of this failure class.

Verdict wording amended: mimi-fp16 is **exonerated on token trajectory,
convicted on onset transient rendering.**

Actions taken: --mimi-fp16 removed from the --fast preset (remains
opt-in), --fast rebenchmarked, fast clips regenerated with the new
preset (the previous fp16-mimi clips are kept as fast-mimifp16-s*.wav
for the record).
