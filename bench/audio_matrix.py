# SPDX-License-Identifier: MIT
"""Multi-seed listening matrix generator.

For each quantization scheme, loads the models ONCE and generates one 30s
output WAV per seed (fresh streaming session + prompt phase per seed),
mirroring offline.py's run_inference flow. Output: <out>/<scheme>-s<seed>.wav
plus the sampled text tokens as JSON.

Usage:
  taskset -c 5-9,15-19 python bench/audio_matrix.py \
      --input-wav bench/results/20260723-audio/input-30s.wav \
      --voice-prompt NATF1.pt --seeds 42424,1001,2002 \
      --schemes bf16,w8a16,fp8-depq8,w8a16-nvfp4ffn,fast \
      --out bench/results/20260724-audio-v2
"""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from moshi.models import loaders, LMGen
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.offline import seed_all, wrap_with_system_tags, warmup

DEFAULT_PROMPT = ("You are a wise and friendly teacher. Answer questions or "
                  "provide advice in a clear and engaging way.")

SCHEMES = {
    # name: (quantize_fn(lm), dep_q_exit, skip_other, mimi_fp16)
    "bf16": (None, None, False, False),
    "w8a16": ("w8a16", None, False, False),
    "fp8-depq8": ("fp8", 8, False, False),
    "w8a16-nvfp4ffn": ("nvfp4+w8a16", None, False, False),
    "fast": ("w8a16", 8, True, True),
}


def log(m):
    print(f"[matrix] {m}", flush=True)


def quantize(lm, kind):
    if kind is None:
        return
    if kind == "fp8":
        from moshi.fp8_quantize import quantize_model
        quantize_model(lm)
    elif kind == "w8a16":
        from moshi.w8a16_quantize import quantize_model_w8a16
        quantize_model_w8a16(lm)
    elif kind == "nvfp4+w8a16":
        from moshi.nvfp4_quantize import quantize_model_nvfp4
        from moshi.w8a16_quantize import quantize_model_w8a16
        quantize_model_nvfp4(lm, scope="ffn")
        quantize_model_w8a16(lm)
    else:
        raise ValueError(kind)


def run_scheme(scheme, args, voice_prompt_path, text_tokenizer, out_dir):
    qkind, depq, skip_other, mimi_fp16 = SCHEMES[scheme]
    device = args.device
    log(f"=== scheme {scheme}: loading models ===")
    mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = None if skip_other else loaders.get_mimi(mimi_weight, device)
    if mimi_fp16:
        mimi = mimi.half()
        mimi.torch_compile_encoder_decoder = True
        if other_mimi is not None:
            other_mimi = other_mimi.half()
            other_mimi.torch_compile_encoder_decoder = True

    moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()
    quantize(lm, qkind)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.8, temp_text=0.7, top_k=250, top_k_text=25,
        depformer_early_exit=depq,
    )
    mimi.streaming_forever(1)
    if other_mimi is not None:
        other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    class _NoopMimi:
        def encode(self, c): return None
        def decode(self, c): return None
        def reset_streaming(self): pass
    om = other_mimi if other_mimi is not None else None

    with torch.no_grad():
        # warmup once per scheme (offline.py warmup handles other_mimi=None
        # via the PR3-style guards; replicate inline to stay standalone)
        wdtype = next(mimi.parameters()).dtype
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=wdtype, device=device)
            codes = mimi.encode(chunk)
            if om is not None:
                _ = om.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = mimi.decode(tokens[:, 1:9])
                if om is not None:
                    _ = om.decode(tokens[:, 1:9])
        torch.cuda.synchronize()

        if str(voice_prompt_path).endswith(".pt"):
            lm_gen.load_voice_prompt_embeddings(str(voice_prompt_path))
        else:
            lm_gen.load_voice_prompt(str(voice_prompt_path))
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            wrap_with_system_tags(args.text_prompt))

        sample_rate = mimi.sample_rate
        user_audio = lm_load_audio(args.input_wav, sample_rate)
        total_target = user_audio.shape[-1]

        for seed in [int(s) for s in args.seeds.split(",")]:
            log(f"--- {scheme} seed {seed} ---")
            seed_all(seed)
            mimi.reset_streaming()
            if om is not None:
                om.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            frames, texts = [], []
            for enc in lm_encode_from_sphn(
                    mimi,
                    lm_iterate_audio(user_audio,
                                     sample_interval_size=lm_gen._frame_size,
                                     pad=True),
                    max_batch=1):
                for c in range(enc.shape[-1]):
                    tokens = lm_gen.step(enc[:, :, c: c + 1])
                    if tokens is None:
                        continue
                    pcm = mimi.decode(tokens[:, 1:9])
                    if om is not None:
                        _ = om.decode(tokens[:, 1:9])
                    frames.append(pcm.float().detach().cpu().numpy()[0, 0])
                    tt = tokens[0, 0, 0].item()
                    if tt not in (0, 3):
                        texts.append(
                            text_tokenizer.id_to_piece(tt).replace("▁", " "))
            out = np.concatenate(frames, axis=-1)[:total_target]
            if out.shape[-1] < total_target:
                out = np.concatenate(
                    [out, np.zeros(total_target - out.shape[-1],
                                   dtype=out.dtype)])
            wav = out_dir / f"{scheme}-s{seed}.wav"
            sphn.write_wav(str(wav), out.astype(np.float32), sample_rate)
            (out_dir / f"{scheme}-s{seed}.json").write_text(
                json.dumps(texts, ensure_ascii=False))
            log(f"wrote {wav} ({''.join(texts)[:80]!r}...)")

    del lm, lm_gen, mimi, other_mimi
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--voice-prompt", default="NATF1.pt")
    ap.add_argument("--seeds", default="42424,1001,2002")
    ap.add_argument("--schemes", default="bf16,w8a16,fp8-depq8,w8a16-nvfp4ffn,fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import os
    try:
        os.sched_setaffinity(0, {5, 6, 7, 8, 9, 15, 16, 17, 18, 19})
    except OSError:
        pass

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    voices_tgz = Path(hf_hub_download(args.hf_repo, "voices.tgz"))
    vdir = voices_tgz.parent / "voices"
    if not vdir.exists():
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    vp = vdir / args.voice_prompt
    assert vp.exists(), vp

    tok_path = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tok_path))

    for scheme in [s.strip() for s in args.schemes.split(",")]:
        run_scheme(scheme, args, vp, text_tokenizer, out_dir)
    log("matrix complete")


if __name__ == "__main__":
    main()
