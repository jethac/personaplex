# SPDX-License-Identifier: MIT
"""Headless per-frame latency benchmark for PersonaPlex on GB10.

Implements the pre-registered protocol in bench/protocol.md. Reproduces the
steady-state per-frame work of moshi/moshi/server.py's opus_loop:

    [opus roundtrip] -> mimi.encode (+other_mimi.encode) -> lm_gen.step
                     -> mimi.decode (+other_mimi.decode) -> D2H copy

with deterministic seeded input, CUDA-synchronized per-stage wall timers,
NVTX ranges for Nsight profiling, per-frame CSV output and a percentile
summary. Never starts the webserver.

Usage (see protocol.md section 8):
    taskset -c 5-9,15-19 python bench/bench.py --frames 500 --out bench/results/label
    taskset -c 5-9,15-19 python bench/bench.py --sustained-minutes 10 --out bench/results/label
    python bench/bench.py --smoke --frames 50 --out /tmp/smoke   # no model/GPU needed
"""

import argparse
import contextlib
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEED = 42424
FRAME_BUDGET_MS = 80.0
WARM_SKIP_DEFAULT = 25
PERF_CORES = "5-9,15-19"  # Cortex-X925 cores on the GB10 ThinkStation PGX
DEFAULT_TEXT_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)

try:  # NVTX is a no-op shim when CUDA is unavailable (smoke mode on CPU)
    torch.cuda.nvtx.range_push("bench_init")
    torch.cuda.nvtx.range_pop()
    nvtx_push, nvtx_pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
except Exception:  # noqa: BLE001
    nvtx_push, nvtx_pop = (lambda _msg: None), (lambda: None)

STAGES = [
    "opus_ms",
    "encode_main_ms",
    "encode_other_ms",
    "step_ms",
    "decode_main_ms",
    "decode_other_ms",
    "d2h_ms",
]
CSV_HEADER = ["frame_idx", "t_wall_s"] + STAGES + ["total_ms", "warm"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(msg: str):
    print(f"[bench] {msg}", flush=True)


def parse_cores(spec: str):
    cores = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cores.update(range(int(a), int(b) + 1))
        elif part:
            cores.add(int(part))
    return cores


def pin_cores(spec: str):
    if spec.lower() == "none":
        log("CPU pinning disabled")
        return None
    cores = parse_cores(spec)
    try:
        os.sched_setaffinity(0, cores)
        log(f"pinned to CPUs {sorted(cores)}")
        return sorted(cores)
    except OSError as e:
        log(f"WARNING: could not set affinity: {e}")
        return None


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def nvsmi(query: str, what: str = "--query-gpu"):
    try:
        out = subprocess.run(
            ["nvidia-smi", what + "=" + query, "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001 - env capture is best-effort
        return f"unavailable: {e}"


def git_info(repo_root: Path):
    def _run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def collect_env(args, pinned):
    repo_root = Path(__file__).resolve().parent.parent
    env = {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git": git_info(repo_root),
        "cpu_affinity": pinned,
        "argv": sys.argv[1:],
        "seed": args.seed,
        "frame_budget_ms": FRAME_BUDGET_MS,
    }
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["sm"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        env["torch_cuda"] = torch.version.cuda
    env["nvidia_smi"] = {
        "driver_gpu": nvsmi("driver_version,name,pstate,temperature.gpu,"
                            "clocks.sm,clocks.mem,power.draw"),
        "compute_apps": nvsmi("pid,process_name,used_memory",
                              "--query-compute-apps"),
    }
    return env


@contextlib.contextmanager
def dmon_capture(path: Path, enabled: bool):
    """Run `nvidia-smi dmon` alongside the measurement, per protocol.md #7."""
    proc = None
    if enabled:
        try:
            f = open(path, "w")
            proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pucm", "-o", "DT"],
                stdout=f, stderr=subprocess.STDOUT,
            )
            log(f"nvidia-smi dmon capture -> {path}")
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: dmon capture unavailable: {e}")
            proc = None
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def nvtx_wrap_method(obj, name: str, label: str):
    """Wrap a bound method in an NVTX range (no-op cost without a profiler)."""
    orig = getattr(obj, name, None)
    if orig is None:
        return
    def wrapped(*a, **k):
        nvtx_push(label)
        try:
            return orig(*a, **k)
        finally:
            nvtx_pop()
    setattr(obj, name, wrapped)


class StageTimer:
    """Wall timer with optional CUDA sync at stage boundaries (protocol #5)."""

    def __init__(self, sync: bool):
        self.sync = sync and torch.cuda.is_available()
        self.times = {}

    def _now(self):
        if self.sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    @contextlib.contextmanager
    def stage(self, name: str):
        nvtx_push(name)
        t0 = self._now()
        try:
            yield
        finally:
            t1 = self._now()
            nvtx_pop()
            self.times[name] = self.times.get(name, 0.0) + (t1 - t0) * 1e3


# --------------------------------------------------------------------------
# deterministic input signal (protocol #3)
# --------------------------------------------------------------------------

def build_input_pattern(sample_rate: int, seed: int) -> np.ndarray:
    """3-second repeating pattern: 1s silence, 1s 440Hz sine, 1s seeded noise."""
    rng = np.random.default_rng(seed)
    sr = sample_rate
    silence = np.zeros(sr, dtype=np.float32)
    t = np.arange(sr, dtype=np.float32) / sr
    sine = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = (0.05 * rng.standard_normal(sr)).astype(np.float32)
    return np.concatenate([silence, sine, noise])


class FrameFeeder:
    def __init__(self, pattern: np.ndarray, frame_size: int, device: str):
        self.pattern = pattern
        self.frame_size = frame_size
        self.device = device
        self.pos = 0

    def next_frame(self) -> torch.Tensor:
        n, fs = len(self.pattern), self.frame_size
        idx = (self.pos + np.arange(fs)) % n
        self.pos = (self.pos + fs) % n
        chunk = self.pattern[idx]
        return torch.from_numpy(chunk).to(self.device).view(1, 1, fs)


# --------------------------------------------------------------------------
# smoke-mode stubs (validate harness without weights/GPU)
# --------------------------------------------------------------------------

class _StubMimi:
    sample_rate = 24000
    frame_rate = 12.5

    def __init__(self, device):
        self.device = device

    def encode(self, chunk):
        time.sleep(0.001)
        return torch.randint(0, 2048, (1, 8, 1), device=self.device)

    def decode(self, codes):
        time.sleep(0.001)
        return torch.zeros(1, 1, 1920, device=self.device)

    def streaming_forever(self, bs):
        pass

    def reset_streaming(self):
        pass


class _StubLMGen:
    def __init__(self, device):
        self.device = device
        self._frame_size = 1920

    def step(self, codes):
        time.sleep(0.005)
        return torch.randint(0, 2048, (1, 9, 1), device=self.device)

    def streaming_forever(self, bs):
        pass


# --------------------------------------------------------------------------
# model setup (mirrors moshi/moshi/offline.py)
# --------------------------------------------------------------------------

def load_models(args, cold: dict):
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders, LMGen

    t0 = time.perf_counter()
    mimi_weight = args.mimi_weight or hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, args.device)
    other_mimi = None if args.skip_other_mimi else loaders.get_mimi(mimi_weight, args.device)
    if args.mimi_fp16:
        mimi = mimi.half()
        mimi.torch_compile_encoder_decoder = True  # unlocks torch_compile_lazy
        if other_mimi is not None:
            other_mimi = other_mimi.half()
            other_mimi.torch_compile_encoder_decoder = True
    cold["load_mimi_s"] = time.perf_counter() - t0
    log(f"mimi loaded in {cold['load_mimi_s']:.1f}s")

    t0 = time.perf_counter()
    tokenizer_path = args.tokenizer or hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    import sentencepiece
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    cold["load_tokenizer_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    moshi_weight = args.moshi_weight or hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=args.device)
    lm.eval()
    cold["load_lm_s"] = time.perf_counter() - t0
    log(f"moshi LM loaded in {cold['load_lm_s']:.1f}s")

    if args.fp8:
        from moshi.fp8_quantize import quantize_model
        t0 = time.perf_counter()
        quantize_model(lm)
        cold["fp8_quantize_s"] = time.perf_counter() - t0
        log(f"FP8 quantization done in {cold['fp8_quantize_s']:.1f}s")
    elif args.w8a16:
        from moshi.w8a16_quantize import quantize_model_w8a16
        t0 = time.perf_counter()
        quantize_model_w8a16(lm)
        cold["w8a16_quantize_s"] = time.perf_counter() - t0
        log(f"w8a16 quantization done in {cold['w8a16_quantize_s']:.1f}s")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=args.device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=not args.greedy,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.topk_audio,
        top_k_text=args.topk_text,
        depformer_early_exit=args.dep_q_exit if args.dep_q_exit > 0 else None,
    )
    mimi.streaming_forever(1)
    if other_mimi is not None:
        other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # NVTX on LMGen sub-stages so nsys traces split forward/depformer/sampling
    for meth, label in [
        ("prepare_step_input", "lm.prepare_step_input"),
        ("process_transformer_output", "lm.process_transformer_output"),
        ("depformer_step", "lm.depformer_step"),
    ]:
        nvtx_wrap_method(lm_gen, meth, label)

    # warmup, identical to server.py / offline.py
    t0 = time.perf_counter()
    with torch.no_grad():
        wdtype = torch.float16 if args.mimi_fp16 else torch.float32
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=wdtype, device=args.device)
            codes = mimi.encode(chunk)
            if other_mimi is not None:
                _ = other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = mimi.decode(tokens[:, 1:9])
                if other_mimi is not None:
                    _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold["warmup_s"] = time.perf_counter() - t0
    log(f"warmup done in {cold['warmup_s']:.2f}s")

    if args.fp8:
        from moshi.fp8_quantize import free_bf16_inproj
        free_bf16_inproj(lm)

    # prompt phase (cold start; excluded from sustained loop)
    t0 = time.perf_counter()
    if not args.no_voice_prompt:
        vp_path = _resolve_voice_prompt(args)
        if vp_path is None:
            log("voice prompt unavailable; continuing without prompts")
            cold["voice_prompt"] = None
        else:
            with torch.no_grad():
                if str(vp_path).endswith(".pt"):
                    lm_gen.load_voice_prompt_embeddings(str(vp_path))
                else:
                    lm_gen.load_voice_prompt(str(vp_path))
                lm_gen.text_prompt_tokens = text_tokenizer.encode(
                    f"<system> {args.text_prompt.strip()} <system>")
                mimi.reset_streaming()
                if other_mimi is not None:
                    other_mimi.reset_streaming()
                lm_gen.reset_streaming()
                lm_gen.step_system_prompts(mimi)
                mimi.reset_streaming()
            cold["voice_prompt"] = str(vp_path)
    else:
        cold["voice_prompt"] = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold["prompt_phase_s"] = time.perf_counter() - t0
    log(f"prompt phase done in {cold['prompt_phase_s']:.2f}s")

    return mimi, other_mimi, lm_gen, text_tokenizer, frame_size


def _resolve_voice_prompt(args):
    try:
        from huggingface_hub import hf_hub_download
        import tarfile
        if args.voice_prompt_dir:
            vdir = Path(args.voice_prompt_dir)
        else:
            tgz = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
            vdir = tgz.parent / "voices"
            if not vdir.exists():
                with tarfile.open(tgz, "r:gz") as tar:
                    tar.extractall(path=tgz.parent)
        path = vdir / args.voice_prompt
        return path if path.exists() else None
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: voice prompt resolution failed: {e}")
        return None


# --------------------------------------------------------------------------
# benchmark loop
# --------------------------------------------------------------------------

def run_loop(args, mimi, other_mimi, lm_gen, frame_size, out_dir: Path):
    device = args.device
    feeder = FrameFeeder(build_input_pattern(getattr(mimi, "sample_rate", 24000),
                                            args.seed), frame_size, device)

    opus_writer = opus_reader = None
    if args.opus:
        import sphn
        sr = getattr(mimi, "sample_rate", 24000)
        opus_writer = sphn.OpusStreamWriter(sr)
        opus_reader = sphn.OpusStreamReader(sr)

    if args.sustained_minutes > 0:
        deadline = time.perf_counter() + args.sustained_minutes * 60.0
        max_frames = 10**9
        log(f"sustained run: {args.sustained_minutes} min")
    else:
        deadline = None
        max_frames = args.frames
        log(f"fixed run: {max_frames} frames")

    csv_path = out_dir / "frames.csv"
    rows = []
    csv_f = open(csv_path, "w")
    csv_f.write(",".join(CSV_HEADER) + "\n")

    t_start = time.perf_counter()
    frame_idx = 0
    profiling = False
    with torch.no_grad():
        while frame_idx < max_frames:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if args.profile_frames > 0:
                if frame_idx == args.warm_skip:
                    torch.cuda.cudart().cudaProfilerStart()
                    profiling = True
                    log(f"cudaProfilerStart at frame {frame_idx}")
                elif profiling and frame_idx == args.warm_skip + args.profile_frames:
                    torch.cuda.cudart().cudaProfilerStop()
                    profiling = False
                    log(f"cudaProfilerStop at frame {frame_idx}")
            timer = StageTimer(sync=not args.no_stage_sync)
            nvtx_push(f"frame_{frame_idx}")
            t_frame0 = timer._now()

            chunk = feeder.next_frame()
            if args.mimi_fp16:
                chunk = chunk.half()

            if opus_writer is not None:
                # Simulates the server's opus input path cost. The opus stream
                # has algorithmic latency, so early frames may return no PCM;
                # we always feed the original chunk to mimi below.
                with timer.stage("opus_ms"):
                    opus_writer.append_pcm(chunk.view(-1).cpu().numpy())
                    payload = opus_writer.read_bytes()
                    if payload:
                        opus_reader.append_bytes(payload)
                        _ = opus_reader.read_pcm()

            with timer.stage("encode_main_ms"):
                codes = mimi.encode(chunk)
            if other_mimi is not None:
                with timer.stage("encode_other_ms"):
                    _ = other_mimi.encode(chunk)

            tokens = None
            with timer.stage("step_ms"):
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c: c + 1])

            if tokens is not None:
                with timer.stage("decode_main_ms"):
                    pcm = mimi.decode(tokens[:, 1:9])
                if other_mimi is not None:
                    with timer.stage("decode_other_ms"):
                        _ = other_mimi.decode(tokens[:, 1:9])
                with timer.stage("d2h_ms"):
                    _ = pcm.detach().cpu().numpy()

            t_frame1 = timer._now()
            nvtx_pop()

            total_ms = (t_frame1 - t_frame0) * 1e3
            warm = 1 if frame_idx >= args.warm_skip else 0
            row = {
                "frame_idx": frame_idx,
                "t_wall_s": round(t_frame1 - t_start, 6),
                **{s: round(timer.times.get(s, 0.0), 4) for s in STAGES},
                "total_ms": round(total_ms, 4),
                "warm": warm,
            }
            rows.append(row)
            csv_f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")
            if frame_idx % 50 == 0:
                csv_f.flush()
                log(f"frame {frame_idx}: total={total_ms:.1f}ms "
                    f"(step={timer.times.get('step_ms', 0.0):.1f}ms)")
            frame_idx += 1
    csv_f.close()
    log(f"wrote {csv_path} ({frame_idx} frames)")
    return rows


def pctl(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def summarize(rows, cold, args):
    warm_rows = [r for r in rows if r["warm"] == 1]
    use = warm_rows if warm_rows else rows
    summary = {
        "config": "smoke" if args.smoke else args.config_label,
        "frames_total": len(rows),
        "frames_summarized": len(use),
        "warm_skip": args.warm_skip,
        "cold_start": cold,
        "stages": {},
    }
    for key in ["total_ms"] + STAGES:
        vals = [r[key] for r in use]
        summary["stages"][key] = {
            "mean": round(float(np.mean(vals)), 3) if vals else None,
            "p50": round(pctl(vals, 50), 3),
            "p90": round(pctl(vals, 90), 3),
            "p95": round(pctl(vals, 95), 3),
            "p99": round(pctl(vals, 99), 3),
            "p99.9": round(pctl(vals, 99.9), 3),
            "max": round(max(vals), 3) if vals else None,
        }
    totals = [r["total_ms"] for r in use]
    misses = [t >= FRAME_BUDGET_MS for t in totals]
    streak = best = 0
    for m in misses:
        streak = streak + 1 if m else 0
        best = max(best, streak)
    summary["budget"] = {
        "budget_ms": FRAME_BUDGET_MS,
        "miss_count": int(sum(misses)),
        "miss_fraction": round(float(np.mean(misses)), 5) if misses else None,
        "longest_miss_streak": best,
    }
    return summary


def format_summary(summary):
    lines = []
    lines.append(f"config: {summary['config']}")
    lines.append(f"frames: {summary['frames_summarized']} summarized "
                 f"(of {summary['frames_total']}, warm_skip={summary['warm_skip']})")
    cold = summary["cold_start"]
    if cold:
        cs = ", ".join(f"{k}={v:.2f}s" if isinstance(v, float) else f"{k}={v}"
                       for k, v in cold.items())
        lines.append(f"cold start: {cs}")
    b = summary["budget"]
    lines.append(f"budget {b['budget_ms']:.0f}ms: {b['miss_count']} misses "
                 f"({(b['miss_fraction'] or 0) * 100:.2f}%), "
                 f"longest streak {b['longest_miss_streak']}")
    hdr = f"{'stage':<18}{'mean':>9}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'p99.9':>9}{'max':>9}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for key in ["total_ms"] + STAGES:
        s = summary["stages"][key]
        lines.append(f"{key:<18}" + "".join(
            f"{(s[c] if s[c] is not None else float('nan')):>9.2f}"
            for c in ["mean", "p50", "p90", "p95", "p99", "p99.9", "max"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=500,
                   help="number of frames for a fixed-length run (default 500)")
    p.add_argument("--sustained-minutes", type=float, default=0,
                   help="run for N minutes instead of --frames (protocol: 10, soak: 30)")
    p.add_argument("--out", type=str, default=None,
                   help="output directory (default bench/results/<date>-<config>)")
    p.add_argument("--config-label", type=str, default="bf16-stock",
                   help="label describing the model config under test")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--warm-skip", type=int, default=WARM_SKIP_DEFAULT,
                   help="initial frames excluded from summary percentiles")
    p.add_argument("--pin-cores", type=str, default=PERF_CORES,
                   help=f"CPU list to pin to (default {PERF_CORES}; 'none' disables)")
    p.add_argument("--no-stage-sync", action="store_true",
                   help="only sync at frame boundaries (see protocol.md #5 caveat)")
    p.add_argument("--opus", action="store_true",
                   help="include sphn opus round-trip of the input frame")
    p.add_argument("--profile-frames", type=int, default=0,
                   help="bracket N warm frames with cudaProfilerStart/Stop "
                        "(for nsys --capture-range=cudaProfilerApi)")
    p.add_argument("--no-dmon", action="store_true",
                   help="skip the nvidia-smi dmon side capture")
    p.add_argument("--smoke", action="store_true",
                   help="run with stub models (no weights/GPU) to validate the harness")
    # model / sampling (mirror offline.py defaults)
    p.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1")
    p.add_argument("--moshi-weight", type=str, default=None)
    p.add_argument("--mimi-weight", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--voice-prompt", type=str, default="NATM1.pt")
    p.add_argument("--voice-prompt-dir", type=str, default=None)
    p.add_argument("--no-voice-prompt", action="store_true")
    p.add_argument("--text-prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    p.add_argument("--temp-audio", type=float, default=0.8)
    p.add_argument("--temp-text", type=float, default=0.7)
    p.add_argument("--topk-audio", type=int, default=250)
    p.add_argument("--topk-text", type=int, default=25)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--fp8", action="store_true",
                   help="quantize LM linears to FP8 (torch._scaled_mm path)")
    p.add_argument("--w8a16", action="store_true",
                   help="weight-only 8-bit: fp8-stored weights dequantized "
                        "in a Triton GEMV, all compute bf16 (activations "
                        "untouched); mutually exclusive with --fp8")
    p.add_argument("--mimi-fp16", action="store_true",
                   help="run both mimi instances in fp16 and enable their "
                        "torch.compile path (amarrmb recipe)")
    p.add_argument("--skip-other-mimi", action="store_true",
                   help="skip the second mimi stream. Safe for latency work: "
                        "every other_mimi.encode/decode result in server.py "
                        "is assigned to _ and discarded (lines 123/129/225/"
                        "232 at commit 3428dfd), and its streaming state "
                        "feeds nothing else")
    p.add_argument("--dep-q-exit", type=int, default=0,
                   help="stop the depformer after N steps (>=8; codebooks "
                        "beyond N are provided-side and unused in serve flow)")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.fp8 and args.w8a16:
        p.error("--fp8 and --w8a16 are mutually exclusive")
    if args.smoke and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    pinned = pin_cores(args.pin_cores)
    seed_all(args.seed)

    if args.out:
        out_dir = Path(args.out)
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = "smoke" if args.smoke else args.config_label
        out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}-{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out_dir}")

    env = collect_env(args, pinned)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    cold = {}
    if args.smoke:
        log("SMOKE MODE: stub models, timings are meaningless")
        mimi = _StubMimi(args.device)
        other_mimi = _StubMimi(args.device)
        lm_gen = _StubLMGen(args.device)
        frame_size = 1920
        cold["smoke"] = True
    else:
        t0 = time.perf_counter()
        mimi, other_mimi, lm_gen, _tok, frame_size = load_models(args, cold)
        cold["total_cold_start_s"] = time.perf_counter() - t0

    with dmon_capture(out_dir / "dmon.log",
                      enabled=not args.no_dmon and not args.smoke
                      and torch.cuda.is_available()):
        rows = run_loop(args, mimi, other_mimi, lm_gen, frame_size, out_dir)

    summary = summarize(rows, cold, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = format_summary(summary)
    (out_dir / "summary.txt").write_text(text + "\n")
    print()
    print(text)


if __name__ == "__main__":
    main()
