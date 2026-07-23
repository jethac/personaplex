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


def run_seed(seed, args, mimi, gen_ref, quant_gens, dec_mimis, frame_size, out_dir):
    """quant_gens: list of (name, LMGen); dec_mimis: {name: MimiModel}."""
    log(f"=== seed {seed} ===")
    seed_all(seed)
    mimi.reset_streaming()
    gen_ref.reset_streaming()
    for _, g in quant_gens:
        g.reset_streaming()
    for m in dec_mimis.values():
        m.reset_streaming()

    # identical starting RNG for every model
    seed_all(seed); rng_ref = RngStream()
    rngs = {}
    for name, _ in quant_gens:
        seed_all(seed); rngs[name] = RngStream()

    pattern = build_input_pattern(mimi.sample_rate, seed)
    pos = 0

    names = [n for n, _ in quant_gens]
    csv_path = out_dir / f"seed{seed}-frames.csv"
    f = open(csv_path, "w")
    cols = ["frame"]
    for n in names:
        cols += [f"{n}_diverged", f"{n}_text_l2", f"{n}_text_maxabs",
                 f"{n}_audio_l2", f"{n}_audio_maxabs",
                 f"{n}_rms", f"{n}_click", f"{n}_centroid_hz", f"{n}_flatness"]
    f.write(",".join(cols) + "\n")

    diverged_at = {n: None for n in names}
    diverge_channel = {n: None for n in names}
    deltas = {n: [] for n in names}      # (frame, text_l2, audio_l2)
    health = {n: [] for n in names}      # (frame, rms, click, cen, flat)
    prev_tail = {n: 0.0 for n in names}
    t0 = time.perf_counter()

    with torch.no_grad():
        for fr in range(args.frames):
            idx = (pos + np.arange(frame_size)) % len(pattern)
            pos = (pos + frame_size) % len(pattern)
            chunk = torch.from_numpy(pattern[idx]).to(args.device).view(1, 1, frame_size)
            codes = mimi.encode(chunk)
            step_in = codes[:, :, 0:1]

            out_ref = rng_ref.run(lambda: gen_ref.step(step_in))
            outs = {}
            for name, g in quant_gens:
                outs[name] = rngs[name].run(lambda g=g: g.step(step_in))

            tok_r, logits_r = out_ref
            row = {c: "" for c in cols[1:]}
            if tok_r is None:
                f.write(f"{fr}," + ",".join(row[c] for c in cols[1:]) + "\n")
                continue

            lt_r, la_r = logits_r
            for name, _ in quant_gens:
                tok_q, logits_q = outs[name]
                if diverged_at[name] is None:
                    same = torch.equal(tok_r[0, :9, 0], tok_q[0, :9, 0])
                    lt_q, la_q = logits_q
                    dt = lt_r.float() - lt_q.float()
                    da = la_r[:, :8].float() - la_q[:, :8].float()
                    row[f"{name}_text_l2"] = f"{dt.norm().item():.4f}"
                    row[f"{name}_text_maxabs"] = f"{dt.abs().max().item():.4f}"
                    row[f"{name}_audio_l2"] = f"{da.norm().item():.4f}"
                    row[f"{name}_audio_maxabs"] = f"{da.abs().max().item():.4f}"
                    deltas[name].append((fr, float(row[f"{name}_text_l2"]),
                                         float(row[f"{name}_audio_l2"])))
                    if not same:
                        mism = (tok_r[0, :9, 0] != tok_q[0, :9, 0]).nonzero().flatten().tolist()
                        diverged_at[name] = fr
                        diverge_channel[name] = mism
                        log(f"seed {seed}: {name} first divergence at frame {fr}, channels {mism}")
                row[f"{name}_diverged"] = "1" if diverged_at[name] is not None else "0"

                pcm = dec_mimis[name].decode(tok_q[:, 1:9])
                x = pcm.float().detach().cpu().numpy()[0, 0]
                rms = float(np.sqrt(np.mean(x * x)))
                click = float(abs(x[0] - prev_tail[name]))
                prev_tail[name] = float(x[-1])
                cen, flat = spectral_stats(x)
                row[f"{name}_rms"] = f"{rms:.5f}"
                row[f"{name}_click"] = f"{click:.5f}"
                row[f"{name}_centroid_hz"] = f"{cen:.1f}"
                row[f"{name}_flatness"] = f"{flat:.5f}"
                health[name].append((fr, rms, click, cen, flat))

            f.write(f"{fr}," + ",".join(row[c] for c in cols[1:]) + "\n")
            if fr % 200 == 0:
                f.flush()
                el = time.perf_counter() - t0
                log(f"seed {seed} frame {fr} ({el:.0f}s, diverged={diverged_at})")
    f.close()

    def drift_fit(series):
        pts = [(fr, v) for fr, v in series if fr >= 2 and v > 0]
        if len(pts) < 10:
            return None, None
        lx = np.log([p[0] for p in pts]); ly = np.log([p[1] for p in pts])
        slope, _ = np.polyfit(lx, ly, 1)
        return float(slope), len(pts)

    summary = {"seed": seed, "frames": args.frames, "models": {}}
    for name in names:
        d = deltas[name]
        exp_audio, n_pts = drift_fit([(fr, a) for fr, _, a in d])
        exp_text, _ = drift_fit([(fr, t) for fr, t, _ in d])
        h = np.array([(r, c) for _, r, c, _, _ in health[name]])
        rms_arr, click_arr = h[:, 0], h[:, 1]
        sil = rms_arr < SILENCE_RMS
        best = cur = 0
        for sflag in sil:
            cur = cur + 1 if sflag else 0
            best = max(best, cur)
        summary["models"][name] = {
            "first_divergence_frame": diverged_at[name],
            "divergence_channels": diverge_channel[name],
            "pre_divergence_frames": len(d),
            "drift_exponent_audio_l2": exp_audio,
            "drift_exponent_text_l2": exp_text,
            "drift_points": n_pts,
            "delta_first10_audio_l2_mean": float(np.mean([x[2] for x in d[:10]])) if d else None,
            "delta_last10_audio_l2_mean": float(np.mean([x[2] for x in d[-10:]])) if d else None,
            "health": {
                "rms_mean": float(rms_arr.mean()),
                "silence_frac": float(sil.mean()),
                "longest_silence_run_frames": int(best),
                "click_p99": float(np.percentile(click_arr, 99)),
                "click_max": float(click_arr.max()),
                "clicks_over_thresh": int((click_arr > CLICK_THRESH).sum()),
                "centroid_hz_mean": float(np.mean([c for *_, c, _ in health[name]])),
                "flatness_mean": float(np.mean([fl for *_, fl in health[name]])),
            },
        }
        log(f"seed {seed} {name}: div@{diverged_at[name]} "
            f"exp_audio={exp_audio} health={json.dumps(summary['models'][name]['health'])}")
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
        elif scheme == "nvfp4ffn-fp8":
            from moshi.nvfp4_quantize import quantize_model_nvfp4
            quantize_model_nvfp4(lm_q, scope="ffn")
            quantize_model(lm_q)
        elif scheme == "nvfp4ffn-w8a16":
            from moshi.nvfp4_quantize import quantize_model_nvfp4
            quantize_model_nvfp4(lm_q, scope="ffn")
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
