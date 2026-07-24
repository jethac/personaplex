# PersonaPlex GB10 Benchmark Protocol (pre-registered)

This document fixes the measurement protocol for all PersonaPlex latency work on
the Lenovo ThinkStation PGX (NVIDIA GB10, sm_121, aarch64). It is written
*before* optimization work begins; all baseline and optimized runs must follow
it so numbers are comparable. Any deviation must be recorded in the results
directory of the affected run.

## 1. Objective and budget

PersonaPlex is full-duplex: it must produce one 80 ms audio frame every 80 ms
(12.5 Hz frame rate, 1920 samples @ 24 kHz). The metric that matters is
**per-frame wall latency of the full serve-path step**; a frame is *on budget*
iff its total step time < 80 ms. We track tail percentiles, not just the mean,
because a single late frame is an audible glitch.

## 2. Measured pipeline (parity with `moshi/moshi/server.py`)

One benchmark "frame" reproduces the steady-state per-frame work of the real
server loop (`ServerState.handle_conn.opus_loop`):

1. *(optional, `--opus`)* Opus round-trip of the user PCM frame through
   `sphn.OpusStreamWriter`/`OpusStreamReader` (the server receives Opus).
2. `mimi.encode(chunk)` — user audio to codes (**and** `other_mimi.encode`,
   as the server does for the loopback stream).
3. `lm_gen.step(codes)` — temporal transformer forward, depformer steps,
   text+audio sampling (sub-stages exposed via NVTX ranges, see §6).
4. `mimi.decode(tokens[:, 1:9])` — agent codes to PCM (**and**
   `other_mimi.decode`, as the server does).
5. Device-to-host copy of the decoded PCM (`.cpu().numpy()`).

Prompt machinery (voice prompt + text prompt injection via
`step_system_prompts`) runs once at session start and is timed separately as
part of cold start, never inside the sustained loop.

## 3. Input: fixed, seeded, deterministic

- Sampling seed: **42424** (torch, CUDA, numpy, python `random`), applied with
  the same `seed_all` strategy as `offline.py`/`server.py`.
- User audio is generated deterministically at 24 kHz from the seed, as a
  repeating 3-second pattern: 1 s silence, 1 s 440 Hz sine at 0.1 amplitude,
  1 s seeded Gaussian noise at 0.05 RMS (band-limited by construction of the
  frame). This exercises silence and non-silence code paths without shipping
  audio files.
- Text prompt: the repo default teacher prompt. Voice prompt: `NATM1.pt` (or
  `--voice-prompt`); if voice assets are unavailable the run records
  `voice_prompt: none` and skips prompt injection (still comparable between
  runs that use the same setting).
- Sampling params: repo defaults (temp_audio 0.8, temp_text 0.7, topk_audio
  250, topk_text 25), sampling enabled. Note: sampled *values* may still
  diverge across code changes; the protocol fixes the *workload*, not the
  token trajectory.

## 4. Runs

Every benchmark session consists of:

1. **Cold start run**: from process start — model load, warmup (4 iterations,
   identical to `server.py` warmup), prompt phase. Each timed and reported
   separately (`cold_start` block in summary).
2. **Sustained run**: default **10 minutes** (`--sustained-minutes 10`,
   configurable up to 30 for soak tests; short functional runs may use
   `--frames N`, e.g. 500 frames = 40 s of audio). The first 25 frames of the
   sustained loop are recorded but flagged `warm=0` and excluded from summary
   percentiles (CUDA graph / allocator settling).

## 5. Reported metrics

Per run, from per-frame wall timers (CUDA-synchronized at stage boundaries):

- **p50 / p90 / p95 / p99 / p99.9 / max / mean** of total frame time (ms),
  and the same percentiles per stage (opus, encode_main, encode_other, step,
  decode_main, decode_other, d2h).
- **Budget misses**: count and fraction of frames with total ≥ 80 ms; longest
  consecutive miss streak.
- **Time series CSV** (`frames.csv`): one row per frame:
  `frame_idx, t_wall_s, opus_ms, encode_main_ms, encode_other_ms, step_ms,
  decode_main_ms, decode_other_ms, d2h_ms, total_ms, warm`.
- **Summary** (`summary.json` + human-readable `summary.txt` table).

Caveat (pre-registered): per-stage timing inserts `torch.cuda.synchronize()`
between stages, which can add small overhead vs. a free-running loop. The
total is therefore an upper bound; `--no-stage-sync` mode times only the whole
frame (single sync) for cross-checking. Both modes' totals must be reported if
they differ by >2 ms at p50.

## 6. Profiling hooks

All stages and LMGen sub-stages (`prepare_step_input`, main transformer
forward, `process_transformer_output` incl. depformer + sampling,
`depformer_step`) are wrapped in `torch.cuda.nvtx.range(...)` so the same
binary can run under Nsight Systems (`nsys profile`) without modification.
NVTX ranges are always on; they are no-ops without a profiler attached.

## 7. Environment block (recorded with every run)

Saved as `env.json` next to the CSV:

- Hostname, date (UTC), git commit of the repo (and dirty flag).
- Driver version, CUDA runtime version, GPU name (from `nvidia-smi`).
- torch / triton versions, python version.
- CPU affinity used (default: performance cores 5-9,15-19, Cortex-X925).
- GPU clocks at run start (`nvidia-smi --query-gpu=clocks.sm,clocks.mem`),
  power state, temperature.
- Concurrent GPU processes (`nvidia-smi --query-compute-apps=...`) — the PGX
  is a shared box; co-tenants must be recorded, and heavy co-tenant activity
  invalidates a run.
- A `nvidia-smi dmon -s pucm` capture runs alongside the sustained run,
  saved as `dmon.log` (1 s cadence), to catch thermal/clock excursions.

## 8. Execution

```
cd ~/work/personaplex
source ~/work/venv-pp/bin/activate
taskset -c 5-9,15-19 python bench/bench.py \
    --frames 500 --out bench/results/<label>  # short run
taskset -c 5-9,15-19 python bench/bench.py \
    --sustained-minutes 10 --out bench/results/<label>  # protocol run
```

`bench.py` also sets `os.sched_setaffinity` to the same cores as a backstop
(`--pin-cores` to override, `--pin-cores none` to disable). Headless only:
never benchmark through the SSL webserver. One run at a time on the box.

## 9. Results layout

```
bench/results/<label>/
  frames.csv      # per-frame time series
  summary.json    # percentiles, budget misses, cold-start block
  summary.txt     # human-readable table
  env.json        # environment block
  dmon.log        # nvidia-smi dmon capture (sustained runs)
```

`<label>` convention: `YYYYMMDD-<shortsha>-<config>`, e.g.
`20260723-f2e0698-bf16-stock`.
