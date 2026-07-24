# SPDX-License-Identifier: MIT
"""Weight-only 8-bit quantization (w8a16) scaffolding for PersonaPlex/Moshi.

Weights are stored as float8_e4m3fn and dequantized before a bf16 matmul —
activation precision is untouched (unlike the FP8 path, which also
quantizes activations for torch._scaled_mm).

Layer selection and forward-patching scaffolding (module walk,
min_features gate, depformer-self_attn skip, bare in_proj handling,
class-level gating/attention forward patches) is derived from
fp8_quantize.py by amarrmb (github.com/amarrmb/personaplex), adapted to
dispatch multiple quantization schemes per instance.

This commit uses a naive dequantize-then-cuBLAS linear as a correctness
baseline; the fused Triton dequant-in-register GEMV and per-channel
scaling land in the follow-up commit.
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def w8a16_linear(x, w_fp8, scale, bias=None):
    """Weight-only-8bit linear: dequantize to bf16, then cuBLAS.

    Naive correctness baseline — streams 1 byte/weight from memory only
    after the dequant materialization, so it is NOT faster than bf16 yet;
    the fused Triton GEMV replaces this in the next commit.
    """
    w = (w_fp8.to(torch.float32) * scale).to(x.dtype)
    return F.linear(x, w, bias)


# ============================================================================
# Forward patches (dispatch on instance flags; fp8-compatible, see module doc)
# ============================================================================

def _w8a16_forward(self, x):
    return w8a16_linear(x, self.weight, self.w8a16_scale, self.bias)


def _make_gating_forward():
    from .fp8_quantize import fp8_linear

    def _apply(lin, x):
        # Per-instance dispatch: this class-level patch may be installed
        # after fp8_quantize's, so it must recognize both schemes' markers.
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


def _make_attn_forward():
    from einops import rearrange as _rearrange
    from .fp8_quantize import fp8_linear

    def attn_forward(self, query, key, value):
        import moshi.modules.transformer as tf_mod

        state = self._streaming_state
        T = query.shape[1]
        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            offset = state.offset
            offset_cpu = state.offset_cpu

        if self.weights_per_step:
            projected = tf_mod.multi_linear(
                self.weights_per_step, self.in_proj_weight, query, offset_cpu
            )
        elif getattr(self, "_in_proj_w8a16", False):
            projected = w8a16_linear(query, self._in_proj_q_weight,
                                     self._in_proj_q_scale)
        elif getattr(self, "_in_proj_fp8", False):
            projected = fp8_linear(query, self._in_proj_fp8_weight,
                                   self._in_proj_scale)
        else:
            projected = F.linear(query, self.in_proj_weight)

        q, k, v = _rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(
                T, device=query.device, dtype=torch.long
            ).view(-1, 1)
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = _rearrange(x, "b h t d -> b t (h d)")
        if self.weights_per_step:
            x = tf_mod.multi_linear(
                self.weights_per_step, self.out_proj.weight, x, offset_cpu
            )
        else:
            out_proj = self.out_proj
            if getattr(out_proj, "_is_w8a16", False):
                x = w8a16_linear(x, out_proj.weight, out_proj.w8a16_scale)
            elif getattr(out_proj, "_is_fp8", False):
                x = fp8_linear(x, out_proj.weight, out_proj.weight_scale)
            else:
                x = out_proj(x)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x

    return attn_forward


# ============================================================================
# Weight quantization
# ============================================================================

def _quantize_weight(w):
    """Per-tensor fp8e4m3 storage (as in fp8_quantize). Returns
    (w_fp8, scale_fp32 scalar)."""
    amax = w.abs().amax()
    scale = (amax / 448.0).clamp(min=1e-12).float().view(1)
    w_fp8 = (w.float() / scale).to(torch.float8_e4m3fn)
    return w_fp8, scale


def quantize_linear_w8a16(module):
    w_fp8, scale = _quantize_weight(module.weight.data)
    module.weight = nn.Parameter(w_fp8, requires_grad=False)
    module.register_buffer("w8a16_scale", scale)
    module._is_w8a16 = True
    module.forward = types.MethodType(_w8a16_forward, module)


def quantize_model_w8a16(model, min_features=512):
    """Quantize all large Linear layers to weight-only fp8 (bf16 compute).

    Coverage mirrors fp8_quantize.quantize_model: skips small linears and
    depformer self_attn; also converts main-attention bare in_proj_weight.
    """
    import moshi.modules.gating as gating_mod
    import moshi.modules.transformer as tf_mod

    gating_mod.ActivationGating.forward = _make_gating_forward()
    tf_mod.StreamingMultiheadAttention.forward = _make_attn_forward()

    n_lin = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if module.in_features < min_features and module.out_features < min_features:
                continue
            if module.weight.ndim > 2:
                continue
            if "depformer" in name and "self_attn" in name:
                continue
            quantize_linear_w8a16(module)
            n_lin += 1

    n_inproj = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, tf_mod.StreamingMultiheadAttention):
            if "depformer" in name or module.weights_per_step:
                continue
            w = module.in_proj_weight.data
            if w.ndim != 2:
                continue
            w_fp8, scale = _quantize_weight(w)
            module.register_buffer("_in_proj_q_weight", w_fp8)
            module.register_buffer("_in_proj_q_scale", scale)
            module._in_proj_w8a16 = True
            # free the bf16 copy immediately (kernel never reads it)
            module.in_proj_weight = nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=w.device),
                requires_grad=False,
            )
            n_inproj += 1

    torch.cuda.empty_cache()
    logger.info(f"[w8a16] Quantized {n_lin} Linear + {n_inproj} in_proj_weight")
    logger.info(f"[w8a16] GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model
