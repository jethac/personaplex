# SPDX-License-Identifier: MIT
"""GEMV bandwidth micro-lab for GB10 (sm_121, LPDDR5X ~273 GB/s peak).

Standalone probe that sweeps batch-1 GEMV load-pipeline variants at the exact
shapes that dominate the PersonaPlex LM step (see bench/AUDIT.md), and emits
a variant -> effective-GB/s tuning table. Effective bandwidth counts weight
bytes only (weights dwarf activations at these shapes).

Axes swept (Triton):
  - BLOCK_OUT x BLOCK_IN tile shape (controls per-CTA burst length and
    concurrency; BLOCK_IN x elem_size is the per-row contiguous burst),
  - num_warps (memory-level parallelism inside a CTA),
  - num_stages (software pipelining of the K loop = cp.async staging),
  - cache_modifier on weight loads ("" vs .cg streaming vs .ca),
  - split-K across SMs (grid axis 1 + fp32 atomic reduction),
  - weight layout: row-major vs burst-tiled arena ([O/BO, K/BI, BO, BI]
    contiguous tiles in exact access order).

References measured alongside: cuBLAS bf16 gemv (F.linear), fp8
torch._scaled_mm, the shipped w8a16/nvfp4 Triton kernels.

Usage:
  taskset -c 5-9,15-19 python bench/microbench/gemv_lab.py --quick
  taskset -c 5-9,15-19 python bench/microbench/gemv_lab.py --full \
      --out bench/microbench/results-gb10.csv
"""

import argparse
import itertools
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SHAPES = [
    (4096, 12288),   # attention in_proj
    (4096, 16896),   # gating linear_in
    (8448, 4096),    # gating linear_out
    (4096, 4096),    # attention out_proj
    (1024, 4224),    # depformer gating in
    (4096, 32000),   # text head
]


def bench(fn, n=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


# ----------------------------------------------------------------------------
# variant kernels
# ----------------------------------------------------------------------------

@triton.jit
def _gemv_rowmajor(x_ptr, w_ptr, y_ptr, IN: tl.constexpr, OUT,
                   BO: tl.constexpr, BI: tl.constexpr, CM: tl.constexpr):
    pid = tl.program_id(0)
    offs_o = pid * BO + tl.arange(0, BO)
    mask_o = offs_o < OUT
    acc = tl.zeros((BO,), dtype=tl.float32)
    for i in range(0, IN, BI):
        offs_i = i + tl.arange(0, BI)
        mask_i = offs_i < IN
        x = tl.load(x_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        if CM == 1:
            w = tl.load(w_ptr + offs_o[:, None] * IN + offs_i[None, :],
                        mask=mask_o[:, None] & mask_i[None, :], other=0.0,
                        cache_modifier=".cg").to(tl.float32)
        elif CM == 2:
            w = tl.load(w_ptr + offs_o[:, None] * IN + offs_i[None, :],
                        mask=mask_o[:, None] & mask_i[None, :], other=0.0,
                        cache_modifier=".ca").to(tl.float32)
        else:
            w = tl.load(w_ptr + offs_o[:, None] * IN + offs_i[None, :],
                        mask=mask_o[:, None] & mask_i[None, :],
                        other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(y_ptr + offs_o, acc, mask=mask_o)


@triton.jit
def _gemv_tiled(x_ptr, w_ptr, y_ptr, IN: tl.constexpr, OUT: tl.constexpr,
                BO: tl.constexpr, BI: tl.constexpr):
    # w laid out as [OUT//BO, IN//BI, BO, BI] contiguous tiles: every CTA
    # reads one fully-sequential arena stripe (burst-aligned).
    pid = tl.program_id(0)
    offs_o = pid * BO + tl.arange(0, BO)
    mask_o = offs_o < OUT
    acc = tl.zeros((BO,), dtype=tl.float32)
    NTI: tl.constexpr = (IN + BI - 1) // BI
    base = pid * NTI * BO * BI
    for t in range(NTI):
        offs_i = t * BI + tl.arange(0, BI)
        mask_i = offs_i < IN
        x = tl.load(x_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + base + t * BO * BI
                    + tl.arange(0, BO)[:, None] * BI
                    + tl.arange(0, BI)[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(y_ptr + offs_o, acc, mask=mask_o)


@triton.jit
def _gemv_splitk(x_ptr, w_ptr, y_ptr, IN: tl.constexpr, OUT,
                 BO: tl.constexpr, BI: tl.constexpr, SK: tl.constexpr):
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    mask_o = offs_o < OUT
    span = (IN + SK - 1) // SK
    k0 = pid_k * span
    acc = tl.zeros((BO,), dtype=tl.float32)
    for i in range(0, span, BI):
        offs_i = k0 + i + tl.arange(0, BI)
        mask_i = offs_i < tl.minimum(k0 + span, IN)
        x = tl.load(x_ptr + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_o[:, None] * IN + offs_i[None, :],
                    mask=mask_o[:, None] & mask_i[None, :],
                    other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.atomic_add(y_ptr + offs_o, acc, mask=mask_o)


def run_variant(kind, w8, x, OUT, IN, BO, BI, warps, stages, cm=0, sk=1,
                w_tiled=None):
    y = torch.zeros(OUT, device="cuda", dtype=torch.float32)
    if kind == "row":
        grid = (triton.cdiv(OUT, BO),)
        fn = lambda: _gemv_rowmajor[grid](x, w8, y, IN=IN, OUT=OUT, BO=BO,
                                          BI=BI, CM=cm, num_warps=warps,
                                          num_stages=stages)
    elif kind == "tiled":
        grid = (triton.cdiv(OUT, BO),)
        fn = lambda: _gemv_tiled[grid](x, w_tiled, y, IN=IN, OUT=OUT, BO=BO,
                                       BI=BI, num_warps=warps,
                                       num_stages=stages)
    elif kind == "splitk":
        grid = (triton.cdiv(OUT, BO), sk)
        def fn():
            y.zero_()
            _gemv_splitk[grid](x, w8, y, IN=IN, OUT=OUT, BO=BO, BI=BI, SK=sk,
                               num_warps=warps, num_stages=stages)
    return bench(fn)


def make_tiled(w8, BO, BI):
    OUT, IN = w8.shape
    O2, I2 = triton.cdiv(OUT, BO) * BO, triton.cdiv(IN, BI) * BI
    wp = torch.zeros(O2, I2, device="cuda", dtype=w8.dtype)
    wp[:OUT, :IN] = w8
    t = wp.view(O2 // BO, BO, I2 // BI, BI).permute(0, 2, 1, 3).contiguous()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0)
    print(f"# GEMV micro-lab on {dev}, torch {torch.__version__}")
    rows = []

    def record(shape, variant, sec, bytes_):
        gbps = bytes_ / sec / 1e9
        rows.append((f"{shape[0]}x{shape[1]}", variant, sec * 1e6, gbps))
        print(f"{shape[0]:>5}x{shape[1]:<6} {variant:<44} "
              f"{sec*1e6:8.1f} us {gbps:7.1f} GB/s")

    # sweep space
    if args.full:
        BOs, BIs, WPs, STs = [8, 16, 32, 64], [256, 512, 1024], [1, 2, 4, 8], [1, 2, 3, 4]
    else:
        BOs, BIs, WPs, STs = [16, 32], [512, 1024], [2, 4], [2, 3]

    sweep_shape = (4096, 12288)
    torch.manual_seed(0)

    for IN, OUT in SHAPES:
        wb = (torch.randn(OUT, IN, device="cuda", dtype=torch.bfloat16) * 0.02)
        x = torch.randn(1, IN, device="cuda", dtype=torch.bfloat16)
        wbytes_bf16 = OUT * IN * 2
        wbytes_fp8 = OUT * IN

        # references
        sec = bench(lambda: F.linear(x, wb))
        record((IN, OUT), "cublas-bf16", sec, wbytes_bf16)

        amax = wb.abs().amax()
        ws = (amax / 448.0).clamp(min=1e-12).float().view(1)
        w8 = (wb / ws).to(torch.float8_e4m3fn)
        xs = torch.ones(1, device="cuda")
        x8 = x.to(torch.float8_e4m3fn)
        try:
            sec = bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=xs,
                                                 scale_b=ws,
                                                 out_dtype=torch.bfloat16))
            record((IN, OUT), "torch._scaled_mm-fp8", sec, wbytes_fp8)
        except Exception as e:
            print(f"  _scaled_mm failed: {str(e)[:80]}")

        import sys
        sys.path.insert(0, "/home/jethac/work/personaplex/moshi")
        from moshi.w8a16_quantize import w8a16_linear, _quantize_weight
        wq, sq = _quantize_weight(wb)
        sec = bench(lambda: w8a16_linear(x.view(1, 1, IN), wq, sq))
        record((IN, OUT), "shipped-w8a16-triton", sec, wbytes_fp8)

        # sweep (full grid only on the sweep shape; elsewhere winners only)
        combos = (itertools.product(BOs, BIs, WPs, STs)
                  if (IN, OUT) == sweep_shape
                  else [(16, 512, 2, 2), (16, 1024, 2, 3), (32, 512, 4, 2)])
        best = None
        for BO, BI, WP, ST in combos:
            for cm, cmname in ((0, ""), (1, ".cg")):
                try:
                    sec = run_variant("row", w8, x.view(-1), OUT, IN, BO, BI,
                                      WP, ST, cm=cm)
                except Exception:
                    continue
                name = f"row BO{BO} BI{BI} w{WP} s{ST}{cmname}"
                if best is None or sec < best[0]:
                    best = (sec, name)
                if (IN, OUT) != sweep_shape or args.full:
                    record((IN, OUT), name, sec, wbytes_fp8)
        if best:
            record((IN, OUT), f"BEST-ROW[{best[1]}]", best[0], wbytes_fp8)

        # tiled arena + split-K at representative configs
        for BO, BI in [(16, 512), (32, 1024)]:
            wt = make_tiled(w8, BO, BI)
            try:
                sec = run_variant("tiled", w8, x.view(-1), OUT, IN, BO, BI,
                                  2, 3, w_tiled=wt)
                record((IN, OUT), f"tiled-arena BO{BO} BI{BI} w2 s3", sec,
                       wbytes_fp8)
            except Exception as e:
                print(f"  tiled failed: {str(e)[:80]}")
            del wt
        for sk in (2, 4):
            try:
                sec = run_variant("splitk", w8, x.view(-1), OUT, IN, 16, 512,
                                  2, 3, sk=sk)
                record((IN, OUT), f"splitk{sk} BO16 BI512 w2 s3", sec,
                       wbytes_fp8)
            except Exception as e:
                print(f"  splitk failed: {str(e)[:80]}")
        del wb, w8, wq
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            f.write("shape,variant,us,eff_GBps\n")
            for sh, v, us, g in rows:
                f.write(f"{sh},{v},{us:.1f},{g:.1f}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
