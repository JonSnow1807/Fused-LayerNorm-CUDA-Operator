"""``torch.library.custom_op`` registrations for every fused CUDA op.

Why this layer exists: a direct call into the pybind extension is opaque to
Dynamo and hard-graph-breaks under ``torch.compile(fullgraph=True)``.  Wrapping
each entry point as a custom op (with a fake impl for FakeTensor/AOTAutograd)
makes the fused paths trace as a single graph node.  The public wrappers in
``layernorm.py`` (and the other op modules) route through
``torch.ops.fused_layernorm.*`` uniformly, so the eager and compiled paths
exercise identical code; the raw ``fused_layernorm_cuda`` pybind functions
remain available for callers who want zero dispatcher overhead (the overhead
is measured and published in the benchmarks).

Registration happens unconditionally at import (it does not need the compiled
extension or a GPU); only *calling* an op requires them, and the wrappers'
eligibility checks route those calls away when the extension is missing.

Every op here is CUDA-only, forward-only for now (the ``_fwd_train``/``_bwd``
pairs with ``register_autograd`` land with the backward kernels), and
``mutates_args=()`` unless stated otherwise.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._common import _ext


def _require_ext():
    if _ext is None:  # pragma: no cover - unreachable through the wrappers
        raise RuntimeError(
            "fused_layernorm_cuda extension is not built; "
            "the Python wrappers should have taken the PyTorch fallback"
        )
    return _ext


@torch.library.custom_op("fused_layernorm::layer_norm", mutates_args=(), device_types="cuda")
def layer_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    return _require_ext().layernorm(input, weight, bias, eps)


@layer_norm.register_fake
def _(input, weight=None, bias=None, eps=1e-5):
    # The real op always returns a fresh contiguous tensor of input's shape
    # (bindings flatten to 2-D and view back), so new_empty - not empty_like,
    # which would preserve a non-contiguous input's strides.
    return input.new_empty(input.shape)


@torch.library.custom_op("fused_layernorm::rms_norm", mutates_args=(), device_types="cuda")
def rms_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    # eps is always concrete here: the wrapper resolves the F.rms_norm
    # eps=None machine-epsilon convention before calling.
    return _require_ext().rmsnorm(input, weight, eps)


@rms_norm.register_fake
def _(input, weight=None, eps=1e-6):
    return input.new_empty(input.shape)


# --------------------------------------------------------------------------- #
# Training forwards (return the per-row statistics autograd saves) and the
# backward ops. The backward ops are custom ops themselves so a compiled
# backward graph traces them as single nodes. register_autograd is attached to
# the *_fwd_train ops only; the inference ops above stay forward-only and the
# wrappers pick per call.
#
# Conventions: statistics are acc-dtype (fp32; fp64 for double inputs) with
# the input's leading shape. Unrequested grads come back from the extension as
# empty 0-element tensors (fixed-arity schemas); _grad_or_none maps them for
# the engine. The statistics outputs are marked non-differentiable, so user
# code cannot backprop through them (raises at .backward() time); the zero
# cotangents the engine still materialises for them here are ignored, as
# aten's own native_layer_norm backward does with dmean/drstd.
# --------------------------------------------------------------------------- #


def _grad_or_none(t: Optional[Tensor]) -> Optional[Tensor]:
    return None if t is None or t.numel() == 0 else t


def _needs(ctx, idx: int) -> bool:
    """needs_input_grad entry for input position ``idx``, defensively.

    Tensor inputs come first and in declaration order in every observed
    variant of this API, but whether trailing non-tensor args (eps) get an
    entry differs, so never fixed-length-unpack the tuple.
    """
    nig = ctx.needs_input_grad
    return bool(nig[idx]) if idx < len(nig) else False


# The statistics outputs are marked non-differentiable in every
# setup_context: differentiating through them from user code raises at
# .backward() time, and the engine still materialises ZERO cotangents for
# them here, which the backward functions ignore (as aten's own
# native_layer_norm backward does with dmean/drstd).


@torch.library.custom_op(
    "fused_layernorm::layer_norm_fwd_train", mutates_args=(), device_types="cuda"
)
def layer_norm_fwd_train(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    return _require_ext().layernorm_fwd_train(input, weight, bias, eps)


@layer_norm_fwd_train.register_fake
def _(input, weight=None, bias=None, eps=1e-5):
    acc = torch.float64 if input.dtype == torch.float64 else torch.float32
    stats = input.new_empty(input.shape[:-1], dtype=acc)
    return input.new_empty(input.shape), stats, stats.clone()


@torch.library.custom_op(
    "fused_layernorm::layer_norm_bwd", mutates_args=(), device_types="cuda"
)
def layer_norm_bwd(
    dy: Tensor,
    xz: Tensor,
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor] = None,
    dz_extra: Optional[Tensor] = None,
    need_dx: bool = True,
    need_dgamma: bool = True,
    need_dbeta: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    return _require_ext().layernorm_bwd(
        dy, xz, mean, rstd, weight, dz_extra, need_dx, need_dgamma, need_dbeta
    )


@layer_norm_bwd.register_fake
def _(dy, xz, mean, rstd, weight=None, dz_extra=None, need_dx=True, need_dgamma=True,
      need_dbeta=True):
    n = xz.shape[-1]
    empty0 = xz.new_empty(0)
    dx = xz.new_empty(xz.shape) if need_dx else empty0
    dgamma = xz.new_empty(n) if need_dgamma else empty0.clone()
    dbeta = xz.new_empty(n) if need_dbeta else empty0.clone()
    return dx, dgamma, dbeta


def _layer_norm_setup(ctx, inputs, output):
    input, weight, bias, eps = inputs
    _, mean, rstd = output
    ctx.save_for_backward(input, weight, mean, rstd)
    ctx.has_bias = bias is not None
    ctx.mark_non_differentiable(mean, rstd)


def _layer_norm_backward(ctx, grad_y, grad_mean, grad_rstd):
    input, weight, mean, rstd = ctx.saved_tensors
    need_dx, need_dw, need_db = _needs(ctx, 0), _needs(ctx, 1), _needs(ctx, 2)
    dx, dgamma, dbeta = torch.ops.fused_layernorm.layer_norm_bwd(
        grad_y, input, mean, rstd, weight, None,
        need_dx, need_dw and weight is not None, need_db and ctx.has_bias,
    )
    return _grad_or_none(dx), _grad_or_none(dgamma), _grad_or_none(dbeta), None


torch.library.register_autograd(
    "fused_layernorm::layer_norm_fwd_train",
    _layer_norm_backward,
    setup_context=_layer_norm_setup,
)


@torch.library.custom_op(
    "fused_layernorm::rms_norm_fwd_train", mutates_args=(), device_types="cuda"
)
def rms_norm_fwd_train(
    input: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    return _require_ext().rmsnorm_fwd_train(input, weight, eps)


@rms_norm_fwd_train.register_fake
def _(input, weight=None, eps=1e-6):
    acc = torch.float64 if input.dtype == torch.float64 else torch.float32
    return input.new_empty(input.shape), input.new_empty(input.shape[:-1], dtype=acc)


@torch.library.custom_op(
    "fused_layernorm::rms_norm_bwd", mutates_args=(), device_types="cuda"
)
def rms_norm_bwd(
    dy: Tensor,
    xz: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor] = None,
    dz_extra: Optional[Tensor] = None,
    need_dx: bool = True,
    need_dgamma: bool = True,
) -> tuple[Tensor, Tensor]:
    return _require_ext().rmsnorm_bwd(dy, xz, rstd, weight, dz_extra, need_dx, need_dgamma)


@rms_norm_bwd.register_fake
def _(dy, xz, rstd, weight=None, dz_extra=None, need_dx=True, need_dgamma=True):
    empty0 = xz.new_empty(0)
    dx = xz.new_empty(xz.shape) if need_dx else empty0
    dgamma = xz.new_empty(xz.shape[-1]) if need_dgamma else empty0.clone()
    return dx, dgamma


def _rms_norm_setup(ctx, inputs, output):
    input, weight, eps = inputs
    _, rstd = output
    ctx.save_for_backward(input, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _rms_norm_backward(ctx, grad_y, grad_rstd):
    input, weight, rstd = ctx.saved_tensors
    need_dx, need_dw = _needs(ctx, 0), _needs(ctx, 1)
    dx, dgamma = torch.ops.fused_layernorm.rms_norm_bwd(
        grad_y, input, rstd, weight, None, need_dx, need_dw and weight is not None
    )
    return _grad_or_none(dx), _grad_or_none(dgamma), None


torch.library.register_autograd(
    "fused_layernorm::rms_norm_fwd_train",
    _rms_norm_backward,
    setup_context=_rms_norm_setup,
)


@torch.library.custom_op(
    "fused_layernorm::fused_add_layer_norm_fwd_train", mutates_args=(), device_types="cuda"
)
def fused_add_layer_norm_fwd_train(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _require_ext().fused_add_layernorm_fwd_train(input, residual, weight, bias, eps)


@fused_add_layer_norm_fwd_train.register_fake
def _(input, residual, weight=None, bias=None, eps=1e-5):
    acc = torch.float64 if input.dtype == torch.float64 else torch.float32
    stats = input.new_empty(input.shape[:-1], dtype=acc)
    return (input.new_empty(input.shape), input.new_empty(input.shape), stats, stats.clone())


def _fused_add_ln_setup(ctx, inputs, output):
    input, residual, weight, bias, eps = inputs
    _, z, mean, rstd = output
    # z (the rounded sum) is what the forward normalised - saving an op OUTPUT
    # costs no extra memory, and the backward differentiates exactly what ran.
    ctx.save_for_backward(z, weight, mean, rstd)
    ctx.has_bias = bias is not None
    ctx.mark_non_differentiable(mean, rstd)


def _fused_add_ln_backward(ctx, grad_y, grad_z, grad_mean, grad_rstd):
    z, weight, mean, rstd = ctx.saved_tensors
    need_dx, need_dres = _needs(ctx, 0), _needs(ctx, 1)
    need_dw, need_db = _needs(ctx, 2), _needs(ctx, 3)
    # d/dz of the norm, plus the downstream cotangent of z itself; since
    # z = input + residual, that total IS both input grads.
    dz, dgamma, dbeta = torch.ops.fused_layernorm.layer_norm_bwd(
        grad_y, z, mean, rstd, weight, grad_z,
        need_dx or need_dres, need_dw and weight is not None, need_db and ctx.has_bias,
    )
    dz = _grad_or_none(dz)
    return dz, dz, _grad_or_none(dgamma), _grad_or_none(dbeta), None


torch.library.register_autograd(
    "fused_layernorm::fused_add_layer_norm_fwd_train",
    _fused_add_ln_backward,
    setup_context=_fused_add_ln_setup,
)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_fwd_train", mutates_args=(), device_types="cuda"
)
def fused_add_rms_norm_fwd_train(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    return _require_ext().fused_add_rmsnorm_fwd_train(input, residual, weight, eps)


@fused_add_rms_norm_fwd_train.register_fake
def _(input, residual, weight=None, eps=1e-6):
    acc = torch.float64 if input.dtype == torch.float64 else torch.float32
    return (input.new_empty(input.shape), input.new_empty(input.shape),
            input.new_empty(input.shape[:-1], dtype=acc))


def _fused_add_rms_setup(ctx, inputs, output):
    input, residual, weight, eps = inputs
    _, z, rstd = output
    ctx.save_for_backward(z, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _fused_add_rms_backward(ctx, grad_y, grad_z, grad_rstd):
    z, weight, rstd = ctx.saved_tensors
    need_dx, need_dres, need_dw = _needs(ctx, 0), _needs(ctx, 1), _needs(ctx, 2)
    dz, dgamma = torch.ops.fused_layernorm.rms_norm_bwd(
        grad_y, z, rstd, weight, grad_z, need_dx or need_dres,
        need_dw and weight is not None,
    )
    dz = _grad_or_none(dz)
    return dz, dz, _grad_or_none(dgamma), None


torch.library.register_autograd(
    "fused_layernorm::fused_add_rms_norm_fwd_train",
    _fused_add_rms_backward,
    setup_context=_fused_add_rms_setup,
)


# --------------------------------------------------------------------------- #
# Fused residual-add + norm.
# Pure variants return fresh (out, new_residual); the underscore variants
# mutate `residual` in place (declared via mutates_args so functionalization
# under torch.compile handles them correctly) and return only `out` - the
# wrapper re-packs (out, residual). custom ops may not return aliases of
# their inputs, which is why the mutating variants cannot return the residual.
# --------------------------------------------------------------------------- #


@torch.library.custom_op(
    "fused_layernorm::fused_add_layer_norm", mutates_args=(), device_types="cuda"
)
def fused_add_layer_norm(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    return _require_ext().fused_add_layernorm(input, residual, weight, bias, eps, False)


@fused_add_layer_norm.register_fake
def _(input, residual, weight=None, bias=None, eps=1e-5):
    return input.new_empty(input.shape), input.new_empty(input.shape)


@torch.library.custom_op(
    "fused_layernorm::fused_add_layer_norm_",
    mutates_args={"residual"},
    device_types="cuda",
)
def fused_add_layer_norm_(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    out, _ = _require_ext().fused_add_layernorm(input, residual, weight, bias, eps, True)
    return out


@fused_add_layer_norm_.register_fake
def _(input, residual, weight=None, bias=None, eps=1e-5):
    return input.new_empty(input.shape)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm", mutates_args=(), device_types="cuda"
)
def fused_add_rms_norm(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    return _require_ext().fused_add_rmsnorm(input, residual, weight, eps, False)


@fused_add_rms_norm.register_fake
def _(input, residual, weight=None, eps=1e-6):
    return input.new_empty(input.shape), input.new_empty(input.shape)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_",
    mutates_args={"residual"},
    device_types="cuda",
)
def fused_add_rms_norm_(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    out, _ = _require_ext().fused_add_rmsnorm(input, residual, weight, eps, True)
    return out


@fused_add_rms_norm_.register_fake
def _(input, residual, weight=None, eps=1e-6):
    return input.new_empty(input.shape)


# --------------------------------------------------------------------------- #
# fp8-E4M3 quantised-output RMSNorm ops (inference-only; the wrappers reject
# grad mode before reaching these). Contract: the fp8 bytes equal quantising
# the None-epilogue output with q = round_e4m3(clamp(y * (1/scale), +-448)) -
# note the RECIPROCAL MULTIPLY, not a division (byte-level contract, tested).
# Dynamic scale = clamp(row_amax(|y|), <= scale_ub)/448 with an all-zero-row
# guard, returned with a trailing broadcast dim so out.float() * scale
# dequantises directly.
# --------------------------------------------------------------------------- #


def _fp8_out(input: Tensor) -> Tensor:
    return input.new_empty(input.shape, dtype=torch.float8_e4m3fn)


def _fp8_scale(input: Tensor) -> Tensor:
    return input.new_empty(input.shape[:-1] + (1,), dtype=torch.float32)


@torch.library.custom_op(
    "fused_layernorm::rms_norm_fp8_static", mutates_args=(), device_types="cuda"
)
def rms_norm_fp8_static(
    input: Tensor,
    scale: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    out, _ = _require_ext().rmsnorm_fp8_static(input, scale, weight, eps)
    return out


@rms_norm_fp8_static.register_fake
def _(input, scale, weight=None, eps=1e-6):
    return _fp8_out(input)


@torch.library.custom_op(
    "fused_layernorm::rms_norm_fp8_dynamic", mutates_args=(), device_types="cuda"
)
def rms_norm_fp8_dynamic(
    input: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
    scale_ub: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    return _require_ext().rmsnorm_fp8_dynamic(input, weight, eps, scale_ub)


@rms_norm_fp8_dynamic.register_fake
def _(input, weight=None, eps=1e-6, scale_ub=None):
    return _fp8_out(input), _fp8_scale(input)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_fp8_static", mutates_args=(), device_types="cuda"
)
def fused_add_rms_norm_fp8_static(
    input: Tensor,
    residual: Tensor,
    scale: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    out, z, _ = _require_ext().fused_add_rmsnorm_fp8_static(
        input, residual, scale, weight, eps, False
    )
    return out, z


@fused_add_rms_norm_fp8_static.register_fake
def _(input, residual, scale, weight=None, eps=1e-6):
    return _fp8_out(input), input.new_empty(input.shape)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_fp8_static_",
    mutates_args={"residual"},
    device_types="cuda",
)
def fused_add_rms_norm_fp8_static_(
    input: Tensor,
    residual: Tensor,
    scale: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    out, _, _ = _require_ext().fused_add_rmsnorm_fp8_static(
        input, residual, scale, weight, eps, True
    )
    return out


@fused_add_rms_norm_fp8_static_.register_fake
def _(input, residual, scale, weight=None, eps=1e-6):
    return _fp8_out(input)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_fp8_dynamic", mutates_args=(), device_types="cuda"
)
def fused_add_rms_norm_fp8_dynamic(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
    scale_ub: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    return _require_ext().fused_add_rmsnorm_fp8_dynamic(
        input, residual, weight, eps, scale_ub, False
    )


@fused_add_rms_norm_fp8_dynamic.register_fake
def _(input, residual, weight=None, eps=1e-6, scale_ub=None):
    return _fp8_out(input), input.new_empty(input.shape), _fp8_scale(input)


@torch.library.custom_op(
    "fused_layernorm::fused_add_rms_norm_fp8_dynamic_",
    mutates_args={"residual"},
    device_types="cuda",
)
def fused_add_rms_norm_fp8_dynamic_(
    input: Tensor,
    residual: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 1e-6,
    scale_ub: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    out, _, s = _require_ext().fused_add_rmsnorm_fp8_dynamic(
        input, residual, weight, eps, scale_ub, True
    )
    return out, s


@fused_add_rms_norm_fp8_dynamic_.register_fake
def _(input, residual, weight=None, eps=1e-6, scale_ub=None):
    return _fp8_out(input), _fp8_scale(input)


@torch.library.custom_op(
    "fused_layernorm::layer_norm_gelu", mutates_args=(), device_types="cuda"
)
def layer_norm_gelu(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    approximate: str = "none",
) -> Tensor:
    return _require_ext().layernorm_gelu(input, weight, bias, eps, approximate)


@layer_norm_gelu.register_fake
def _(input, weight=None, bias=None, eps=1e-5, approximate="none"):
    return input.new_empty(input.shape)
