# SPDX-License-Identifier: MIT
"""FP8-vs-bf16 divergence soak for PersonaPlex (M2 tolerance-ladder gate).

Runs the SAME seeded input through a bf16 LM and an FP8-quantized LM in one
process, frame-interleaved, with per-model pinned RNG streams (state swap
around each step), so that:

- both models see identical user-audio codes each frame (shared fp32 mimi
  encoder),
- each model's sampling RNG sequence is identical to what a solo run with
  that seed would consume,
- pre-divergence, sampled-token differences can only be caused by FP8
  numerics flipping a sampling decision, not by RNG desync.

Measures, per seed:
  (a) frame index of first sampled-token divergence (text + 8 agent audio
      codebooks),
  (b) pre-divergence per-frame logit deltas (L2 and max-abs, text + audio)
      and their growth trend (log-log regression exponent: flat/sublinear
      passes, superlinear fails),
  (c) post-divergence reference-free health of the FP8 stream: per-frame
      RMS energy, frame-boundary discontinuity (clicks), silence-run stats,
      spectral centroid/flatness (via FFT), decoded with a streaming mimi.

Outputs per seed: frames CSV; plus a run-level summary.json. Verdict logic
for DIVERGENCE.md: drift exponent < 1 for all seeds = pass.

Usage:
  taskset -c 5-9,15-19 python bench/divergence.py \
      --frames 5000 --seeds 42424,1001,2002,3003,4004 \
      --out bench/results/<date>-divergence
"""

import argparse
import datetime as _dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEEDS = "42424,1001,2002,3003,4004"
SILENCE_RMS = 1e-3
CLICK_THRESH = 0.25  # abs sample jump across frame boundary


def log(msg):
    print(f"[div] {msg}", flush=True)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)


class RngStream:
    """Pinned per-model RNG stream, swapped in around each step."""

    def __init__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state()

    def run(self, fn):
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state(self.cuda)
        out = fn()
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state()
        return out


def build_input_pattern(sample_rate, seed):
    rng = np.random.default_rng(seed)
    sr = sample_rate
    sil = np.zeros(sr, dtype=np.float32)
    t = np.arange(sr, dtype=np.float32) / sr
    sine = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = (0.05 * rng.standard_normal(sr)).astype(np.float32)
    return np.concatenate([sil, sine, noise])


def make_lmgen(lm, mimi, args, LMGen):
    return LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=args.device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.8, temp_text=0.7, top_k=250, top_k_text=25,
        return_logits=True,
    )


def pin_original_forwards(model):
    """Bind the ORIGINAL class forwards onto instances of the classes that
    fp8_quantize.quantize_model patches at class level (ActivationGating,
    StreamingMultiheadAttention). Instance attributes shadow the class
    patch, so the bf16 reference model and mimi keep bit-identical code
    paths even after quantize_model() runs in this process."""
    import types
    import moshi.modules.gating as gating_mod
    import moshi.modules.transformer as tf_mod
    n = 0
    for m in model.modules():
        if isinstance(m, gating_mod.ActivationGating):
            m.forward = types.MethodType(gating_mod.ActivationGating.forward, m)
            n += 1
        elif isinstance(m, tf_mod.StreamingMultiheadAttention):
            m.forward = types.MethodType(tf_mod.StreamingMultiheadAttention.forward, m)
            n += 1
    return n


def spectral_stats(frame_np):
    """Return (centroid_hz, flatness) of one 1920-sample frame @24kHz."""
    w = np.abs(np.fft.rfft(frame_np * np.hanning(len(frame_np))))
    p = w * w + 1e-12
    freqs = np.fft.rfftfreq(len(frame_np), d=1.0 / 24000.0)
    centroid = float((freqs * p).sum() / p.sum())
    flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    return centroid, flatness


def run_seed(seed, args, mimi, gen_bf16, gen_fp8, frame_size, out_dir):
    log(f"=== seed {seed} ===")
    seed_all(seed)
    # fresh streaming sessions (recaptures CUDA graphs)
    mimi.reset_streaming()
    gen_bf16.reset_streaming()
    gen_fp8.reset_streaming()

    # identical starting RNG for both models
    rng_bf16 = RngStream()
    seed_all(seed)
    rng_fp8 = RngStream()

    pattern = build_input_pattern(mimi.sample_rate, seed)
    pos = 0

    csv_path = out_dir / f"seed{seed}-frames.csv"
    f = open(csv_path, "w")
    f.write("frame,diverged,text_l2,text_maxabs,audio_l2,audio_maxabs,"
            "fp8_rms,fp8_click,fp8_centroid_hz,fp8_flatness\n")

    diverged_at = None
    diverge_channel = None
    deltas = []          # (frame, text_l2, audio_l2) pre-divergence
    health = []          # (frame, rms, click, centroid, flatness)
    prev_tail = 0.0
    t0 = time.perf_counter()

    with torch.no_grad():
        for fr in range(args.frames):
            idx = (pos + np.arange(frame_size)) % len(pattern)
            pos = (pos + frame_size) % len(pattern)
            chunk = torch.from_numpy(pattern[idx]).to(args.device).view(1, 1, frame_size)
            codes = mimi.encode(chunk)

            row = {k: "" for k in ["text_l2", "text_maxabs", "audio_l2",
                                   "audio_maxabs", "fp8_rms", "fp8_click",
                                   "fp8_centroid_hz", "fp8_flatness"]}

            for c in range(codes.shape[-1]):
                step_in = codes[:, :, c: c + 1]
                out_b = rng_bf16.run(lambda: gen_bf16.step(step_in))
                out_f = rng_fp8.run(lambda: gen_fp8.step(step_in))
            tok_b, logits_b = out_b
            tok_f, logits_f = out_f
            if tok_b is None or tok_f is None:
                assert tok_b is None and tok_f is None
                f.write(f"{fr},,,,,,,,,\n")
                continue

            if diverged_at is None:
                # compare text + 8 agent audio channels
                same = torch.equal(tok_b[0, :9, 0], tok_f[0, :9, 0])
                lt_b, la_b = logits_b
                lt_f, la_f = logits_f
                dt = (lt_b.float() - lt_f.float())
                da = (la_b[:, :8].float() - la_f[:, :8].float())
                row["text_l2"] = f"{dt.norm().item():.4f}"
                row["text_maxabs"] = f"{dt.abs().max().item():.4f}"
                row["audio_l2"] = f"{da.norm().item():.4f}"
                row["audio_maxabs"] = f"{da.abs().max().item():.4f}"
                deltas.append((fr, float(row["text_l2"]), float(row["audio_l2"])))
                if not same:
                    mism = (tok_b[0, :9, 0] != tok_f[0, :9, 0]).nonzero().flatten().tolist()
                    diverged_at = fr
                    diverge_channel = mism
                    log(f"seed {seed}: first divergence at frame {fr}, channels {mism}")

            # fp8 stream health (decode with shared mimi decoder state)
            pcm = mimi.decode(tok_f[:, 1:9])
            x = pcm.float().detach().cpu().numpy()[0, 0]
            rms = float(np.sqrt(np.mean(x * x)))
            click = float(abs(x[0] - prev_tail))
            prev_tail = float(x[-1])
            cen, flat = spectral_stats(x)
            row["fp8_rms"] = f"{rms:.5f}"
            row["fp8_click"] = f"{click:.5f}"
            row["fp8_centroid_hz"] = f"{cen:.1f}"
            row["fp8_flatness"] = f"{flat:.5f}"
            health.append((fr, rms, click, cen, flat))

            f.write(f"{fr},{1 if diverged_at is not None else 0},"
                    f"{row['text_l2']},{row['text_maxabs']},{row['audio_l2']},"
                    f"{row['audio_maxabs']},{row['fp8_rms']},{row['fp8_click']},"
                    f"{row['fp8_centroid_hz']},{row['fp8_flatness']}\n")
            if fr % 200 == 0:
                f.flush()
                el = time.perf_counter() - t0
                log(f"seed {seed} frame {fr} ({el:.0f}s, "
                    f"diverged={diverged_at})")
    f.close()

    # drift trend: log-log regression of audio_l2 vs frame (skip frame<2)
    def drift_fit(series):
        pts = [(fr, v) for fr, _, v in series if fr >= 2 and v > 0]
        if len(pts) < 10:
            return None, None
        lx = np.log([p[0] for p in pts]); ly = np.log([p[1] for p in pts])
        slope, intercept = np.polyfit(lx, ly, 1)
        return float(slope), len(pts)

    exp_audio, n_pts = drift_fit(deltas)
    text_series = [(fr, 0, t) for fr, t, _ in deltas]
    exp_text, _ = drift_fit(text_series)

    # health aggregates
    h = np.array([(r, c) for _, r, c, _, _ in health])
    rms_arr = h[:, 0]; click_arr = h[:, 1]
    sil = rms_arr < SILENCE_RMS
    # longest silence run
    best = cur = 0
    for s in sil:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    summary = {
        "seed": seed,
        "frames": args.frames,
        "first_divergence_frame": diverged_at,
        "divergence_channels": diverge_channel,
        "pre_divergence_frames": len(deltas),
        "drift_exponent_audio_l2": exp_audio,
        "drift_exponent_text_l2": exp_text,
        "drift_points": n_pts,
        "delta_first10_audio_l2_mean": float(np.mean([d[2] for d in deltas[:10]])) if deltas else None,
        "delta_last10_audio_l2_mean": float(np.mean([d[2] for d in deltas[-10:]])) if deltas else None,
        "fp8_health": {
            "rms_mean": float(rms_arr.mean()),
            "rms_p1": float(np.percentile(rms_arr, 1)),
            "silence_frac": float(sil.mean()),
            "longest_silence_run_frames": int(best),
            "click_p99": float(np.percentile(click_arr, 99)),
            "click_max": float(click_arr.max()),
            "clicks_over_thresh": int((click_arr > CLICK_THRESH).sum()),
            "centroid_hz_mean": float(np.mean([c for *_, c, _ in health])),
            "flatness_mean": float(np.mean([fl for *_, fl in health])),
        },
    }
    log(f"seed {seed} summary: {json.dumps(summary['fp8_health'])}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pin-cores", type=str, default="5-9,15-19")
    p.add_argument("--no-quantize", action="store_true",
                   help="self-test: leave the second LM in bf16 too; the run "
                        "must then show zero divergence and zero logit delta, "
                        "validating the RNG-swap and forward-pinning mechanics")
    args = p.parse_args()

    try:
        cores = set()
        for part in args.pin_cores.split(","):
            a, _, b = part.partition("-")
            cores.update(range(int(a), int(b or a) + 1))
        os.sched_setaffinity(0, cores)
    except OSError:
        pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from moshi.models import loaders, LMGen
    from moshi.fp8_quantize import quantize_model

    log("loading mimi (shared fp32 encoder+decoder)")
    mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, args.device)
    mimi.streaming_forever(1)

    moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    log("loading bf16 LM")
    lm_bf16 = loaders.get_moshi_lm(moshi_weight, device=args.device)
    lm_bf16.eval()
    log("loading second LM copy")
    lm_fp8 = loaders.get_moshi_lm(moshi_weight, device=args.device)
    lm_fp8.eval()
    # Pin original forwards on the reference model and mimi BEFORE
    # quantize_model patches the classes.
    n = pin_original_forwards(lm_bf16) + pin_original_forwards(mimi)
    log(f"pinned original forwards on {n} bf16/mimi modules")
    if args.no_quantize:
        log("SELF-TEST: second LM stays bf16 (expect zero divergence)")
        pin_original_forwards(lm_fp8)
    else:
        quantize_model(lm_fp8)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    gen_bf16 = make_lmgen(lm_bf16, mimi, args, LMGen)
    gen_fp8 = make_lmgen(lm_fp8, mimi, args, LMGen)
    gen_bf16.streaming_forever(1)
    gen_fp8.streaming_forever(1)

    env = {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "frames_per_seed": args.frames,
        "seeds": args.seeds,
        "argv": os.sys.argv[1:],
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    summaries = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        summaries.append(run_seed(seed, args, mimi, gen_bf16, gen_fp8,
                                  frame_size, out_dir))
        (out_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n")
    log("all seeds done")


if __name__ == "__main__":
    main()
