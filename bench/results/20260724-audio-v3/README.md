# Blind mimi-precision A/B set (v3)

Fresh directory, every file generated in one run on 2026-07-24 — nothing
reused from earlier sets (whose identities were muddied by sync timing).

## Design
5 pairs, seeds {42424, 1001, 2002, 7777, 31337}. Within each pair the LM
stack is IDENTICAL (w8a16 + dep-q-exit 8 + skip-other-mimi + rms-fusion);
the only variable is mimi precision: one of a/b is mimi-fp32, the other
mimi-fp16(+compile), randomly assigned per pair (coin flip, secrets
module).

## Listening protocol (honor system)
**Listen BEFORE reading KEY.json.** For each pair, report: which of a/b
(if either) has the softer/degraded onset ("eyy" instead of "hey"), and
any other difference. Then unseal KEY.json and score:
- consistent identification of the fp16 clip across pairs => the onset
  effect is real (demotion stands);
- ~chance performance => reinstate mimi-fp16 into --fast.

## Files
See MANIFEST.md for md5/seed/timestamps. pair<N>-<letter>.json holds the
sampled text tokens of the matching wav.

## RESULT (unsealed 2026-07-24): 0/5 — mimi-fp16 REINSTATED

User's verbatim per-pair results: pair1 both "eyy"; pair2 both "hey" but
pair2-b says "question" not "questions"; pair3 both correct words,
slightly weird intonation; pair4 both "eyy"; pair5 both correct, somewhat
over-enunciated. Within-pair discrimination: **0/5**. Onset character
tracks the SEED (42424/7777 = "eyy" in BOTH precisions; 1001 = "hey" in
both) — a trajectory property, not a precision property. Per the
pre-registered rule, --mimi-fp16 is reinstated into --fast.

### Case study (full episode, recorded honestly)
1. Single-trial listening detected a degraded onset in a fast (fp16-mimi)
   clip vs an isolation (fp32-mimi) clip → mimi-fp16 demoted.
2. The numeric onset analysis had predicted no audible difference
   (0.003 rel-L2 over the shared prefix) — apparently falsified.
3. Sync-timing muddied clip identities; checksums contradicted the
   session; user retracted the comparison.
4. This blind matched-pairs protocol was run: fresh files, sealed key,
   honor-system listen-first.
5. Score 0/5 → the single-trial detection was pattern-matching on
   trajectory differences, not precision; **the numeric onset analysis
   was correct all along.**

Lesson, standing: single-trial listening at threshold pattern-matches
trajectories; blind matched-pairs listening is the reliable perceptual
instrument; numeric prefix analysis and blind listening agreed in the
end.

### pair2 token divergence (expected class)
pair2-a.wav = fast (fp32 mimi), seed 1001, md5 53d4b561...;
pair2-b.wav = fastmimifp16, seed 1001, md5 8afe05f9.... The
"questions"->"question" difference in pair2-b is fp16-encode-induced
token divergence: mimi-fp16 perturbs the encoded user codes, sampling
diverges (the frame-2-3 class documented in DIVERGENCE.md), and the two
clips are different valid trajectories. This is expected and covered by
the drift ladder; it is not a rendering defect.
