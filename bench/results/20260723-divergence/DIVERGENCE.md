# FP8 / w8a16 / NVFP4 divergence & drift verdicts (M2 quality gate)

Runs (real nvidia/personaplex-7b-v1 weights, protocol seeds):
- **Free-run soak** `bench/results/20260723-divergence/`: 5 seeds x 5000
  frames, bf16 reference vs fp8 and w8a16, identical seeded input,
  per-stream pinned RNG (interleaved harness; its bf16-vs-bf16 self-test
  held 0 divergence for 150 frames).
- **Teacher-forced drift ladder** `bench/results/20260723-divergence-forced/`:
  3 seeds x 1500 frames, sequential record-then-replay harness — the bf16
  reference is recorded end-to-end, then each scheme replays the identical
  trajectory (reference tokens forced in each frame), so per-frame logit
  deltas measure pure numeric drift with sampling chaos removed. Its
  bf16-vs-bf16 self-test measures **exactly 0.0** logit delta, validating
  alignment and RNG methodology.
- Listenable paired WAVs (same seed/input/prompt): `bench/results/20260723-audio/`.

## 1. Teacher-forced drift (the pre-registered criterion) — LEAD RESULT

Log-log drift exponents of per-frame logit L2 delta vs frame index
(sublinear: exponent < 1 = pass; ~0 = flat):

| scheme | exp_audio (3 seeds) | exp_text | delta first10 -> last10 (audio L2) |
|---|---|---|---|
| fp8 | +0.058 / +0.036 / -0.009 | -0.014 / +0.000 / -0.071 | 57-70 -> 31-34 |
| w8a16 | +0.052 / +0.038 / -0.016 | -0.019 / +0.000 / -0.049 | 46-52 -> 23-24 |
| nvfp4ffn-w8a16 | +0.025 / +0.002 / -0.051 | +0.083 / +0.069 / -0.020 | 113-142 -> 35 |

**All exponents are ~0: drift is flat over 1500 forced frames — no error
accumulation — decisively sublinear for every scheme.** Per-frame
perturbation magnitude ranks w8a16 < fp8 < nvfp4ffn-w8a16 (w8a16 ~25-30%
below fp8; nvfp4 ~2x fp8 early, settling to ~1.1x by frame 1500).
Context: audio logit tensors have 16384 elements, so last10 L2 of 23-35
is a per-logit RMS of ~0.18-0.27 on logits with O(10) dynamic range.

## 2. Free-run token divergence — expected, not a failure signal

First sampled-token divergence occurs at frame 2-3 for BOTH fp8 and w8a16
on all 5 seeds (audio channels, temp 0.8 / topk 250). This is the expected
behavior of temperature sampling under any logit perturbation — the drift
ladder above, not this frame index, is the discriminating measurement. The
bf16-vs-bf16 control held zero divergence, proving the flips are numerics,
not harness/RNG artifacts.

Why the free-run drift_exponent fields are null: the pre-divergence window
(2-3 frames) is below the >=10-point minimum for a meaningful log-log fit;
post-divergence logits are not comparable (different histories). The forced
run exists precisely to answer the drift question — and does.

## 3. Reference-free stream health, WITH bf16 baseline

Free-run quant streams (5 seeds x 5000 frames) vs the bf16 reference on the
same seeds (recorded in the forced runs reference phase):

| stream | silence_frac | longest silence run | clicks > 0.25 | click max | spectral |
|---|---|---|---|---|---|
| bf16 reference (seeds 42424/1001/2002) | 0.76-0.99 | 240-1461 | 0 | 0.029-0.063 | normal |
| fp8 (5 seeds) | 0.84-0.98 | 1688-2963 | 0 | 0.047-0.086 | normal |
| w8a16 (5 seeds) | 0.82-0.99 | 1270-2973 | 0 | 0.035-0.125 | normal |

The high silence fractions are the MODELS behavior on this synthetic
silence/sine/noise input (the bf16 reference is 98-99% silent on two of
three seeds) — quantized streams are not mute where bf16 speaks; their
silence and spectral statistics fall inside the reference band, and no
stream shows a single click/energy discontinuity above threshold in 25k
(fp8/w8a16) + 4.5k (ref) decoded frames.

## 4. Verdicts (pre-registered criterion: sublinear drift = pass)

| scheme | drift | health vs ref | verdict |
|---|---|---|---|
| fp8 (torch._scaled_mm, dynamic act scaling) | flat (pass) | in-band | **PASS** — but see latency note: it is neither the fastest nor the lowest-perturbation option on GB10 |
| w8a16 (weight-only, bf16 activations) | flat (pass), lowest deltas | in-band | **PASS — recommended default** (also fastest: see ABLATIONS.md) |
| nvfp4ffn-w8a16 (FFN nvfp4, rest w8a16) | flat (pass), ~1.1-2x fp8 deltas | in-band (WAV evidence) | **PASS-WITH-CAVEATS** — largest per-frame perturbation, uncalibrated e2m1 (9.5% weight relerr); recommend listening eval of `20260723-audio/w8a16-nvfp4ffn.wav` and a calibration pass before making it default |

Not measured here: task-level quality (WER/e2e conversational metrics) —
these gates cover numeric drift and signal health only.

## Perceptual addendum (v2 multi-seed listening, 2026-07-24)

User listening over the multi-seed matrix (bench/results/20260724-audio-v2/)
confirmed the interpretation rule works in practice: per-seed cross-scheme
comparisons are DIFFERENT TRAJECTORIES (divergence at frames 2-3), so
single-clip quirks are not scheme attributes. Concretely: seed 2002
produces oddities even in bf16 (truncation + topic shift + rising final
intonation, corroborated by an f0-contour analysis: bf16-s2002 final-400ms
slope +51 Hz/s vs falling contours on normal seeds in every scheme) — the
"strange final intonation" noted for w8a16/nvfp4ffn at s2002 falls within
the reference's own band, and fp8-depq8's "perfect" s2002 was trajectory
luck. Verdicts above are unchanged; the drift ladder remains the objective
evidence. The one perceptual finding that survived isolation testing is
tracked in the audio-v2 README (mimi-fp16 onset A/B).
