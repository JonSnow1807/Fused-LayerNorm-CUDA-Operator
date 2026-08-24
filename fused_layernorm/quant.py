"""fp8-E4M3 quantised-output RMSNorm ops (inference-only).

LLM serving stacks feed RMSNorm outputs straight into fp8 GEMMs; fusing the
quantisation into the norm kernel removes a full read+write of the activation.
Two scaling modes:

* **static** (``scale=`` given): a per-tensor dequant scale — a 1-element fp32
  CUDA tensor, dereferenced on device (no host sync; CUDA-graph capturable).
* **dynamic** (``scale=None``): per-token scales computed in-kernel from the
  row amax (optionally clamped to ``scale_ub``), returned with a trailing
  broadcast dim so ``out.float() * scale`` dequantises directly.

Byte-level contract (tested): the fp8 output equals quantising the plain
``rms_norm`` output with ``round_e4m3(clamp(y * (1/scale), ±448))`` — note the
reciprocal multiply. ``scale`` is the DEQUANT scale (``y ≈ out.float() *
scale``), the vLLM/TensorRT convention.

Everything here is inference-only: calls that would record autograd raise.
Ineligible calls (CPU tensors, odd layouts) run a numerically identical eager
composite so the API works everywhere ``torch.float8_e4m3fn`` does.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ._common import _as_shape, _eligible, _ext, _needs_grad, _resolve_rms_eps, _Shape

__all__ = ["rms_norm_fp8", "fused_add_rms_norm_fp8"]

_FP8_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _check_quant_call(input: torch.Tensor, *tensors, scale, scale_ub) -> None:
    if _needs_grad(input, *tensors, scale):
        raise RuntimeError("fp8 norm ops are inference-only; detach or use no_grad()")
    if scale is not None and scale_ub is not None:
        raise ValueError("scale_ub applies to dynamic scaling only (scale=None)")
    if scale_ub is not None and scale_ub <= 0:
        # The kernel treats <= 0 as "no clamp"; rejecting it here keeps the
        # eager fallback and the kernel path identical.
        raise ValueError(f"scale_ub must be positive, got {scale_ub}")
    if input.dtype not in _FP8_DTYPES:
        raise ValueError(f"fp8 ops support float32/float16/bfloat16 inputs, got {input.dtype}")


def _quantize_ref(y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The byte-contract quantiser (reciprocal multiply, clamp, round)."""
    q = (y.float() * (1.0 / scale)).clamp(-448.0, 448.0)
    return q.to(torch.float8_e4m3fn)


def _dynamic_scale_ref(y: torch.Tensor, scale_ub: Optional[float]) -> torch.Tensor:
    amax = y.float().abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = amax.clamp(max=scale_ub)
    return amax.clamp(min=1e-12) / 448.0


def rms_norm_fp8(
    input: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_ub: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm with fused fp8-E4M3 output. Returns ``(out_fp8, scale)``.

    ``scale`` given → static per-tensor mode (the same tensor is returned);
    ``scale=None`` → dynamic per-token mode. ``eps=None`` follows
    ``F.rms_norm``'s machine-epsilon convention.
    """
    shape = _as_shape(normalized_shape)
    _check_quant_call(input, weight, scale=scale, scale_ub=scale_ub)
    eps_c = _resolve_rms_eps(input.dtype, eps)
    if _eligible(input, shape, weight, ext_available=_ext is not None):
        if not torch.compiler.is_compiling():  # eager fast path (see layer_norm)
            if scale is not None:
                out, _ = _ext.rmsnorm_fp8_static(input, scale, weight, eps_c)
                return out, scale
            return _ext.rmsnorm_fp8_dynamic(input, weight, eps_c, scale_ub)
        if scale is not None:
            out = torch.ops.fused_layernorm.rms_norm_fp8_static(input, scale, weight, eps_c)
            return out, scale
        return torch.ops.fused_layernorm.rms_norm_fp8_dynamic(input, weight, eps_c, scale_ub)
    # Eager composite fallback: numerically identical by construction.
    y = F.rms_norm(input, shape, weight, eps_c)
    s = scale if scale is not None else _dynamic_scale_ref(y, scale_ub)
    return _quantize_ref(y, s), s


def fused_add_rms_norm_fp8(
    input: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_ub: Optional[float] = None,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``new_residual = input + residual; out = fp8(rms_norm(new_residual))``.

    Returns ``(out_fp8, new_residual, scale)``. ``new_residual`` stays in the
    input dtype (it continues the residual stream); only the normalised output
    is quantised. ``inplace=True`` mutates ``residual`` as in
    :func:`fused_layernorm.fused_add_rms_norm`.
    """
    shape = _as_shape(normalized_shape)
    _check_quant_call(input, residual, weight, scale=scale, scale_ub=scale_ub)
    eps_c = _resolve_rms_eps(input.dtype, eps)
    if _eligible(input, shape, residual, weight, ext_available=_ext is not None):
        if not torch.compiler.is_compiling():
            if inplace:
                if scale is not None:
                    out, z, _ = _ext.fused_add_rmsnorm_fp8_static(
                        input, residual, scale, weight, eps_c, True)
                    return out, residual, scale
                return _ext.fused_add_rmsnorm_fp8_dynamic(
                    input, residual, weight, eps_c, scale_ub, True)
            if scale is not None:
                out, z, _ = _ext.fused_add_rmsnorm_fp8_static(
                    input, residual, scale, weight, eps_c, False)
                return out, z, scale
            return _ext.fused_add_rmsnorm_fp8_dynamic(
                input, residual, weight, eps_c, scale_ub, False)
        ops = torch.ops.fused_layernorm
        if inplace:
            if scale is not None:
                out = ops.fused_add_rms_norm_fp8_static_(input, residual, scale, weight, eps_c)
                return out, residual, scale
            out, s = ops.fused_add_rms_norm_fp8_dynamic_(input, residual, weight, eps_c, scale_ub)
            return out, residual, s
        if scale is not None:
            out, z = ops.fused_add_rms_norm_fp8_static(input, residual, scale, weight, eps_c)
            return out, z, scale
        return ops.fused_add_rms_norm_fp8_dynamic(input, residual, weight, eps_c, scale_ub)
    # Eager composite fallback.
    if inplace:
        z = residual.add_(input)
    else:
        z = input + residual
    y = F.rms_norm(z, shape, weight, eps_c)
    s = scale if scale is not None else _dynamic_scale_ref(y, scale_ub)
    return _quantize_ref(y, s), z, s
