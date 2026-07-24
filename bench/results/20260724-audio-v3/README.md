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
