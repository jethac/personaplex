# SPDX-License-Identifier: MIT
"""FP8-vs-bf16 divergence soak for PersonaPlex (M2 tolerance-ladder gate).

Runs the SAME seeded input through a bf16 reference LM and one or more
quantized LMs (full FP8 and/or weight-only w8a16) in one process,
frame-interleaved, with per-model pinned RNG streams (state swap around
each step), so that:

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


def run_reference(seed, args, mimi, gen_ref, frame_size):
    """Phase 1: run the bf16 reference stream end-to-end, recording per frame
    the input codes, output tokens, logits (fp16, CPU) and PCM health."""
    log(f"--- seed {seed}: reference phase ---")
    seed_all(seed)
    mimi.reset_streaming()
    gen_ref.reset_streaming()
    pattern = build_input_pattern(mimi.sample_rate, seed)
    pos = 0
    rec = {"codes": [], "toks": [], "logits": [], "health": []}
    prev_tail = 0.0
    with torch.no_grad():
        for fr in range(args.frames + 1):   # +1: forcing needs toks[fr+1]
            idx = (pos + np.arange(frame_size)) % len(pattern)
            pos = (pos + frame_size) % len(pattern)
            chunk = torch.from_numpy(pattern[idx]).to(args.device).view(1, 1, frame_size)
            codes = mimi.encode(chunk)
            step_in = codes[:, :, 0:1].detach().clone()
            rec["codes"].append(step_in)
            tok, logits = gen_ref.step(step_in)
            if tok is None:
                rec["toks"].append(None)
                rec["logits"].append(None)
                continue
            rec["toks"].append(tok.detach().clone())
            lt, la = logits
            rec["logits"].append((lt.detach().float().cpu(),
                                  la.detach().float().cpu()))
            pcm = mimi.decode(tok[:, 1:9])
            x = pcm.float().detach().cpu().numpy()[0, 0]
            rms = float(np.sqrt(np.mean(x * x)))
            click = float(abs(x[0] - prev_tail))
            prev_tail = float(x[-1])
            cen, flat = spectral_stats(x)
            rec["health"].append((fr, rms, click, cen, flat))
    return rec


def run_scheme(seed, args, name, gen_q, dec_mimi, rec, csv_writer):
    """Phase 2: replay one quantized scheme against the recording.

    Identical seeding to the reference phase, so with --no-quantize the
    stream must be bit-identical (self-test). In forced mode the reference
    frame-fr token set (= rec toks[fr+1], since out at step t carries frame
    t-1 with max_delay=1) is forced in at step fr, keeping trajectories
    identical so logit deltas measure pure numeric drift for the whole run.
    """
    log(f"--- seed {seed}: scheme {name} ({'forced' if args.forced else 'free'}) ---")
    seed_all(seed)
    gen_q.reset_streaming()
    dec_mimi.reset_streaming()
    diverged_at = None
    diverge_channel = None
    deltas = []
    health = []
    prev_tail = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for fr in range(args.frames):
            step_in = rec["codes"][fr]
            if args.forced and fr + 1 < len(rec["toks"]) and rec["toks"][fr + 1] is not None:
                fut = rec["toks"][fr + 1]
                tok, logits = gen_q.step(step_in,
                                         moshi_tokens=fut[:, 1:9],
                                         text_token=fut[:, 0, 0])
            else:
                tok, logits = gen_q.step(step_in)
            tok_r = rec["toks"][fr]
            assert (tok is None) == (tok_r is None), (
                f"{name} emission schedule mismatch at frame {fr}")
            if tok is None:
                csv_writer(fr, name, {})
                continue
            row = {}
            if args.forced or diverged_at is None:
                ref_logits = rec["logits"][fr]
                lt_r = ref_logits[0].to(args.device).float()
                la_r = ref_logits[1].to(args.device).float()
                lt_q, la_q = logits
                dt = lt_r - lt_q.float()
                da = la_r[:, :8] - la_q[:, :8].float()
                row["text_l2"] = f"{dt.norm().item():.4f}"
                row["text_maxabs"] = f"{dt.abs().max().item():.4f}"
                row["audio_l2"] = f"{da.norm().item():.4f}"
                row["audio_maxabs"] = f"{da.abs().max().item():.4f}"
                deltas.append((fr, float(row["text_l2"]), float(row["audio_l2"])))
                if (diverged_at is None and not args.forced
                        and not torch.equal(tok[0, :9, 0], tok_r[0, :9, 0])):
                    mism = (tok[0, :9, 0] != tok_r[0, :9, 0]).nonzero().flatten().tolist()
                    diverged_at = fr
                    diverge_channel = mism
                    log(f"seed {seed}: {name} first divergence at frame {fr}, channels {mism}")
            row["diverged"] = "1" if diverged_at is not None else "0"
            if not args.forced:
                pcm = dec_mimi.decode(tok[:, 1:9])
                x = pcm.float().detach().cpu().numpy()[0, 0]
                rms = float(np.sqrt(np.mean(x * x)))
                click = float(abs(x[0] - prev_tail))
                prev_tail = float(x[-1])
                cen, flat = spectral_stats(x)
                row["rms"] = f"{rms:.5f}"
                row["click"] = f"{click:.5f}"
                row["centroid_hz"] = f"{cen:.1f}"
                row["flatness"] = f"{flat:.5f}"
                health.append((fr, rms, click, cen, flat))
            csv_writer(fr, name, row)
            if fr % 300 == 0:
                log(f"seed {seed} {name} frame {fr} "
                    f"({time.perf_counter()-t0:.0f}s, div={diverged_at})")
    return diverged_at, diverge_channel, deltas, health


def drift_fit(series):
    pts = [(fr, v) for fr, v in series if fr >= 2 and v > 0]
    if len(pts) < 10:
        return None, None
    lx = np.log([p[0] for p in pts]); ly = np.log([p[1] for p in pts])
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope), len(pts)


def health_stats(health):
    if not health:
        return None
    h = np.array([(r, c) for _, r, c, _, _ in health])
    rms_arr, click_arr = h[:, 0], h[:, 1]
    sil = rms_arr < SILENCE_RMS
    best = cur = 0
    for sflag in sil:
        cur = cur + 1 if sflag else 0
        best = max(best, cur)
    return {
        "rms_mean": float(rms_arr.mean()),
        "silence_frac": float(sil.mean()),
        "longest_silence_run_frames": int(best),
        "click_p99": float(np.percentile(click_arr, 99)),
        "click_max": float(click_arr.max()),
        "clicks_over_thresh": int((click_arr > CLICK_THRESH).sum()),
        "centroid_hz_mean": float(np.mean([c for *_, c, _ in health])),
        "flatness_mean": float(np.mean([fl for *_, fl in health])),
    }


def run_seed(seed, args, mimi, gen_ref, quant_gens, dec_mimis, frame_size, out_dir):
    rec = run_reference(seed, args, mimi, gen_ref, frame_size)

    names = [n for n, _ in quant_gens]
    csv_path = out_dir / f"seed{seed}-frames.csv"
    fields = ["diverged", "text_l2", "text_maxabs", "audio_l2", "audio_maxabs",
              "rms", "click", "centroid_hz", "flatness"]
    cols = ["frame"] + [f"{n}_{c}" for n in names for c in fields]
    rows_buf = {}

    def csv_writer(fr, name, row):
        rows_buf.setdefault(fr, {})[name] = row

    summary = {"seed": seed, "frames": args.frames, "forced": args.forced,
               "models": {}}
    summary["reference_health"] = health_stats(rec["health"])

    for name, g in quant_gens:
        diverged_at, ch, deltas, health = run_scheme(
            seed, args, name, g, dec_mimis[name], rec, csv_writer)
        exp_audio, n_pts = drift_fit([(fr, a) for fr, _, a in deltas])
        exp_text, _ = drift_fit([(fr, t) for fr, t, _ in deltas])
        summary["models"][name] = {
            "first_divergence_frame": diverged_at,
            "divergence_channels": ch,
            "compared_frames": len(deltas),
            "drift_exponent_audio_l2": exp_audio,
            "drift_exponent_text_l2": exp_text,
            "drift_points": n_pts,
            "delta_first10_audio_l2_mean": float(np.mean([x[2] for x in deltas[:10]])) if deltas else None,
            "delta_last10_audio_l2_mean": float(np.mean([x[2] for x in deltas[-10:]])) if deltas else None,
            "health": health_stats(health),
        }
        log(f"seed {seed} {name}: div@{diverged_at} exp_audio={exp_audio} "
            f"d10={summary['models'][name]['delta_first10_audio_l2_mean']} "
            f"dlast10={summary['models'][name]['delta_last10_audio_l2_mean']}")

    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for fr in range(args.frames):
            vals = [str(fr)]
            for n in names:
                row = rows_buf.get(fr, {}).get(n, {})
                vals += [row.get(c, "") for c in fields]
            f.write(",".join(vals) + "\n")

    del rec
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pin-cores", type=str, default="5-9,15-19")
    p.add_argument("--schemes", type=str, default="fp8,w8a16",
                   help="comma list of quantization schemes to soak "
                        "against the bf16 reference (fp8, w8a16)")
    p.add_argument("--forced", action="store_true",
                   help="teacher-forced drift mode: the reference model's "
                        "sampled tokens are forced into each quantized model "
                        "every frame, so logit deltas measure pure numeric "
                        "drift on identical trajectories for the whole run "
                        "(free-running divergence/health are skipped)")
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
    from moshi.w8a16_quantize import quantize_model_w8a16

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    log("loading mimi (shared fp32 input encoder)")
    mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, args.device)
    mimi.streaming_forever(1)

    moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    log("loading bf16 reference LM")
    lm_ref = loaders.get_moshi_lm(moshi_weight, device=args.device)
    lm_ref.eval()
    # Pin original forwards on the reference model and mimi BEFORE any
    # quantizer patches the classes.
    n = pin_original_forwards(lm_ref) + pin_original_forwards(mimi)
    log(f"pinned original forwards on {n} bf16/mimi modules")

    quant_lms = []
    for scheme in schemes:
        log(f"loading LM copy for scheme {scheme}")
        lm_q = loaders.get_moshi_lm(moshi_weight, device=args.device)
        lm_q.eval()
        if args.no_quantize:
            log(f"SELF-TEST: {scheme} stays bf16 (expect zero divergence)")
            pin_original_forwards(lm_q)
        elif scheme == "fp8":
            quantize_model(lm_q)
        elif scheme == "w8a16":
            quantize_model_w8a16(lm_q)
        else:
            raise ValueError(f"unknown scheme {scheme}")
        quant_lms.append((scheme, lm_q))

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    gen_ref = make_lmgen(lm_ref, mimi, args, LMGen)
    gen_ref.streaming_forever(1)
    quant_gens = []
    dec_mimis = {}
    for scheme, lm_q in quant_lms:
        g = make_lmgen(lm_q, mimi, args, LMGen)
        g.streaming_forever(1)
        quant_gens.append((scheme, g))
        dm = loaders.get_mimi(mimi_weight, args.device)
        pin_original_forwards(dm)
        dm.streaming_forever(1)
        dec_mimis[scheme] = dm

    env = {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "frames_per_seed": args.frames,
        "seeds": args.seeds,
        "schemes": args.schemes,
        "argv": os.sys.argv[1:],
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    summaries = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        summaries.append(run_seed(seed, args, mimi, gen_ref, quant_gens,
                                  dec_mimis, frame_size, out_dir))
        (out_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n")
    log("all seeds done")


if __name__ == "__main__":
    main()
