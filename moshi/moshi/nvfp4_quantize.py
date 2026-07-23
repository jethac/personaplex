# SPDX-License-Identifier: MIT
"""NVFP4 weight-only quantization for PersonaPlex/Moshi (T2, GB10).

Weights stored as packed e2m1 nibbles with per-16-element e4m3 block scales
and one fp32 per-tensor scale2 (the llama.cpp-dgx / NVFP4 recipe), compute in
bf16/fp32 — activations untouched. The GEMV is a hand-written Triton kernel
that unpacks and dequantizes in registers; at batch=1 decode the linear is
memory-bandwidth-bound, so streaming ~0.56 byte/weight (vs 1.0 fp8, 2.0
bf16) is the win and no tensor-core FP4 MMA is required.

Why not CUTLASS/Marlin (route A): CUTLASS FP4 GEMMs produce corrupt output
on sm_121 (see scottgl9/sglang-spark-gb10-optimizations, which routes
through Marlin with a byte-interleaved scale fix). Transplanting the Marlin
stack into this dependency-light serve stack is disproportionate when the
same Triton GEMV pattern already proven at 8 bit here extends to 4-bit
unpacking (route B, per plan).

Scope control: `quantize_model_nvfp4(model, scope="ffn")` quantizes only the
temporal-transformer FFN (ActivationGating linear_in/linear_out outside the
depformer) — the bulk of weight bytes and the least sensitive layers. The
rest of the model can then be quantized with fp8_quantize.quantize_model
(call nvfp4 FIRST; fp8_quantize skips modules marked `_is_nvfp4`).

The class-level gating forward installed here dispatches nvfp4 / w8a16 /
fp8 / bf16 per instance, superseding and remaining compatible with the
patches from fp8_quantize / w8a16_quantize.
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

logger = logging.getLogger(__name__)

BLOCK = 16  # nvfp4 scale block size (along IN)

# e2m1 magnitude table (code 0..7): 0, .5, 1, 1.5, 2, 3, 4, 6
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ============================================================================
# Triton GEMV over packed nvfp4 weights
# ============================================================================

@triton.jit
def _dec_e2m1(c):
    """Decode a 4-bit e2m1 code tensor (int32) to its float value."""
    sign = 1.0 - 2.0 * ((c >> 3).to(tl.float32))
    e = ((c >> 1) & 3).to(tl.float32)
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0.0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2(e - 1.0))
    return sign * mag


@triton.jit
def _nvfp4_gemv_kernel(
    x_ptr, wp_ptr, bs_ptr, y_ptr,
    scale2,
    IN: tl.constexpr, OUT,
    BLOCK_OUT: tl.constexpr, BLOCK_IN: tl.constexpr,
):
    # wp: [OUT, IN//2] uint8 (lo nibble = even k, hi nibble = odd k)
    # bs: [OUT, IN//16] fp8e4m3 block scales
    pid = tl.program_id(0)
    offs_o = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_o = offs_o < OUT
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    HALF: tl.constexpr = BLOCK_IN // 2
    for i in range(0, IN, BLOCK_IN):
        offs_h = tl.arange(0, HALF)
        offs_b = i // 2 + offs_h                       # packed byte offsets
        mask_b = offs_b < IN // 2
        p = tl.load(wp_ptr + offs_o[:, None] * (IN // 2) + offs_b[None, :],
                    mask=mask_o[:, None] & mask_b[None, :], other=0)
        w_lo = _dec_e2m1((p & 0xF).to(tl.int32))
        w_hi = _dec_e2m1(((p >> 4) & 0xF).to(tl.int32))

        # per-byte block-scale gather (each e4m3 scale covers 8 bytes)
        offs_s = offs_b // 8
        bsx = tl.load(bs_ptr + offs_o[:, None] * (IN // 16) + offs_s[None, :],
                      mask=mask_o[:, None] & mask_b[None, :],
                      other=0.0).to(tl.float32)         # [BO, HALF]

        x_lo = tl.load(x_ptr + i + 2 * offs_h,
                       mask=(i + 2 * offs_h) < IN, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + i + 2 * offs_h + 1,
                       mask=(i + 2 * offs_h + 1) < IN, other=0.0).to(tl.float32)

        acc += tl.sum((w_lo * x_lo[None, :] + w_hi * x_hi[None, :]) * bsx,
                      axis=1)
    y = acc * scale2
    tl.store(y_ptr + offs_o, y.to(y_ptr.dtype.element_ty), mask=mask_o)


def nvfp4_linear(x, w_packed, block_scales, scale2, in_features):
    orig_shape = x.shape
    out_features = w_packed.shape[0]
    x2 = x.reshape(-1, in_features)
    if x2.shape[0] != 1:
        w = dequantize_nvfp4(w_packed, block_scales, scale2, in_features)
        return F.linear(x, w.to(x.dtype)).reshape(*orig_shape[:-1], out_features)
    y = torch.empty(1, out_features, device=x.device, dtype=x.dtype)
    BLOCK_OUT, BLOCK_IN, num_warps = (16, 512, 2) if in_features >= 4096 else (16, 256, 4)
    grid = (triton.cdiv(out_features, BLOCK_OUT),)
    _nvfp4_gemv_kernel[grid](
        x2, w_packed, block_scales, y, scale2,
        IN=in_features, OUT=out_features,
        BLOCK_OUT=BLOCK_OUT, BLOCK_IN=BLOCK_IN, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], out_features)


# ============================================================================
# Host-side quantize / dequantize
# ============================================================================

def quantize_weight_nvfp4(w):
    """w: [OUT, IN] -> (packed uint8 [OUT, IN/2], scales fp8e4m3 [OUT, IN/16],
    scale2 float)."""
    OUT, IN = w.shape
    assert IN % BLOCK == 0, IN
    wf = w.float()
    absmax = wf.abs().amax()
    scale2 = (absmax / (448.0 * 6.0)).clamp(min=1e-12)
    wb = wf.view(OUT, IN // BLOCK, BLOCK)
    bmax = wb.abs().amax(dim=2)
    bscale = (bmax / (6.0 * scale2)).clamp(min=1e-12)
    bscale_fp8 = bscale.to(torch.float8_e4m3fn)
    bscale_eff = bscale_fp8.float().clamp(min=1e-12)
    q = wb / (bscale_eff.unsqueeze(2) * scale2)          # target in [-6, 6]
    mag = q.abs().clamp(max=6.0)
    codes = torch.bucketize(mag, _E2M1_BOUNDS.to(w.device))  # 0..7
    codes = codes | ((q < 0).to(torch.int64) << 3)           # sign bit
    codes = codes.view(OUT, IN).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), bscale_fp8.contiguous(), float(scale2)


def dequantize_nvfp4(packed, block_scales, scale2, in_features):
    OUT = packed.shape[0]
    codes = torch.empty(OUT, in_features, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    vals = _E2M1_VALUES.to(packed.device)[(codes & 7).long()]
    vals = vals * torch.where((codes >> 3) > 0, -1.0, 1.0)
    bs = block_scales.float().repeat_interleave(BLOCK, dim=1)
    return vals * bs * scale2


# ============================================================================
# Module patching
# ============================================================================

def quantize_linear_nvfp4(module):
    packed, bs, s2 = quantize_weight_nvfp4(module.weight.data)
    in_features = module.in_features
    module.weight = nn.Parameter(packed, requires_grad=False)
    module.register_buffer("nvfp4_block_scales", bs)
    module.nvfp4_scale2 = s2
    module.nvfp4_in_features = in_features
    module._is_nvfp4 = True

    def _fwd(self, x):
        return nvfp4_linear(x, self.weight, self.nvfp4_block_scales,
                            self.nvfp4_scale2, self.nvfp4_in_features)
    module.forward = types.MethodType(_fwd, module)


def _make_gating_forward():
    from .fp8_quantize import fp8_linear
    try:
        from .w8a16_quantize import w8a16_linear
    except Exception:  # pragma: no cover
        w8a16_linear = None

    def _apply(lin, x):
        if getattr(lin, "_is_nvfp4", False):
            return nvfp4_linear(x, lin.weight, lin.nvfp4_block_scales,
                                lin.nvfp4_scale2, lin.nvfp4_in_features)
        if getattr(lin, "_is_w8a16", False):
            return w8a16_linear(x, lin.weight, lin.w8a16_scale)
        if getattr(lin, "_is_fp8", False):
            return fp8_linear(x, lin.weight, lin.weight_scale)
        return F.linear(x, lin.weight)

    def gating_forward(self, x):
        x = _apply(self.linear_in, x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        return _apply(self.linear_out, x)

    return gating_forward


def quantize_model_nvfp4(model, scope="ffn"):
    """Apply NVFP4 to the selected scope. scope='ffn': temporal-transformer
    ActivationGating linears only (excludes depformer). Call BEFORE
    fp8_quantize.quantize_model when mixing precisions."""
    import moshi.modules.gating as gating_mod

    assert scope == "ffn", scope
    gating_mod.ActivationGating.forward = _make_gating_forward()

    n = 0
    bytes_before = bytes_after = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, gating_mod.ActivationGating):
            continue
        if "depformer" in name:
            continue
        for lin in (module.linear_in, module.linear_out):
            bytes_before += lin.weight.numel() * lin.weight.element_size()
            quantize_linear_nvfp4(lin)
            bytes_after += (lin.weight.numel()
                            + lin.nvfp4_block_scales.numel())
            n += 1
    torch.cuda.empty_cache()
    logger.info(f"[nvfp4] Quantized {n} FFN linears: "
                f"{bytes_before/1e9:.2f} GB -> {bytes_after/1e9:.2f} GB")
    return model
