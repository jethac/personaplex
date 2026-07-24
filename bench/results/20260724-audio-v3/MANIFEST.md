# MANIFEST — 20260724-audio-v3 (blind mimi-precision A/B set)

Every file below was generated FRESH in this run (2026-07-24), from
the same 30 s deterministic input (md5 of input-30s.wav: dd9e18cf8c60e76c8403c8352d341d9f), voice NATF1.pt,
temp 0.8/0.7, topk 250/25. LM config for ALL clips:
w8a16 + dep-q-exit 8 + skip-other-mimi + rms-fusion. The ONLY
difference within a pair is mimi precision (fp32 vs fp16+compile);
which of a/b is which is randomly assigned per pair and recorded
ONLY in KEY.json (sealed — see README).

| file | md5 | seed | generated | flags |
|---|---|---|---|---|
| pair1-a.wav | 1e1abf2ae5f6d531886fd0b27fe4e549 | 42424 | 2026-07-24T12:53:30+0900 | fast-core; mimi precision per sealed KEY.json |
| pair1-b.wav | dff7bef9ecd9c3c5b5592c6c552f575d | 42424 | 2026-07-24T12:57:14+0900 | fast-core; mimi precision per sealed KEY.json |
| pair2-a.wav | 53d4b561370b679d5943d49d9d86d86c | 1001 | 2026-07-24T12:53:52+0900 | fast-core; mimi precision per sealed KEY.json |
| pair2-b.wav | 8afe05f92d02ae7f0fd17c0b6d6d717e | 1001 | 2026-07-24T12:57:35+0900 | fast-core; mimi precision per sealed KEY.json |
| pair3-a.wav | e95dd9505229811c2fefb215f27c638a | 2002 | 2026-07-24T12:54:14+0900 | fast-core; mimi precision per sealed KEY.json |
| pair3-b.wav | fe3bb8757633c846c5205ae5a3cc4277 | 2002 | 2026-07-24T12:57:56+0900 | fast-core; mimi precision per sealed KEY.json |
| pair4-a.wav | dc6c70ef0a69b49a9431e8ef8444ed03 | 7777 | 2026-07-24T12:58:18+0900 | fast-core; mimi precision per sealed KEY.json |
| pair4-b.wav | 8922cfe359bb87bf96d93306c8bd6a10 | 7777 | 2026-07-24T12:54:36+0900 | fast-core; mimi precision per sealed KEY.json |
| pair5-a.wav | 814604c4fdd848e3f96b0e3d103c3851 | 31337 | 2026-07-24T12:58:39+0900 | fast-core; mimi precision per sealed KEY.json |
| pair5-b.wav | c3ef98b3bc29cc8dbd1ef1998318d466 | 31337 | 2026-07-24T12:54:58+0900 | fast-core; mimi precision per sealed KEY.json |
