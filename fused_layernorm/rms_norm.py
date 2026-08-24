"""RMSNorm: an ``F.rms_norm``-shaped function and an ``nn.RMSNorm`` drop-in.

Same design rules as ``layernorm.py``: the CUDA kernels are used only when
they cannot change behaviour (CUDA input of a supported dtype, 1-D
``normalized_shape`` equal to the last dimension, matching weight
dtype/device, no autocast). Gradient-requiring eligible calls run the
fwd-train kernel with a registered CUDA backward; every other call is exactly
``torch.nn.functional.rms_norm``.

The one subtlety worth knowing: ``F.rms_norm``'s default ``eps=None`` does NOT
mean 1e-5 or 1e-6 — PyTorch substitutes the machine epsilon of the
*computation* dtype (fp32 for fp16/bf16/fp32 inputs, so ~1.19e-7).  This
module replicates that exactly (see ``_common._resolve_rms_eps``), so swapping
``nn.RMSNorm`` for :class:`RMSNorm` changes nothing numerically.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import _eligible, _as_shape, _ext, _replace_modules, _resolve_rms_eps, _Shape

__all__ = ["rms_norm", "RMSNorm", "replace_rmsnorm"]


def rms_norm(
    input: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> torch.Tensor:
    """Root-mean-square normalisation with the same signature as ``F.rms_norm``.

    ``y = x * rsqrt(mean(x^2) + eps) [* weight]`` over the last dimension.
    The fused CUDA kernel is used under the same eligibility rule as
    :func:`fused_layernorm.layer_norm` (see that docstring for the rationale
    of each condition); otherwise this is exactly
    ``torch.nn.functional.rms_norm`` — including ``eps=None`` meaning the
    machine epsilon of the computation dtype, which the fused path resolves
    to the identical concrete value.
    """
    shape = _as_shape(normalized_shape)
    if _eligible(input, shape, weight, ext_available=_ext is not None, needs_grad_ok=True):
        from ._common import _needs_grad

        eps_c = _resolve_rms_eps(input.dtype, eps)
        if _needs_grad(input, weight):
            y, _ = torch.ops.fused_layernorm.rms_norm_fwd_train(input, weight, eps_c)
            return y
        if not torch.compiler.is_compiling():
            return _ext.rmsnorm(input, weight, eps_c)  # eager fast path (see layer_norm)
        return torch.ops.fused_layernorm.rms_norm(input, weight, eps_c)
    return F.rms_norm(input, shape, weight, eps)


class RMSNorm(nn.RMSNorm):
    """``torch.nn.RMSNorm`` whose forward routes through :func:`rms_norm`.

    Subclassing keeps ``__init__``, ``reset_parameters``, ``state_dict``
    layout and ``extra_repr`` identical to ``nn.RMSNorm`` (including
    ``eps=None`` semantics).  Backend selection happens per call inside
    :func:`rms_norm`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)

    @classmethod
    def from_torch(cls, module: nn.RMSNorm) -> "RMSNorm":
        """Build an :class:`RMSNorm` sharing ``module``'s parameters.

        The *same* ``weight`` Parameter object is attached (not a clone), so
        optimisers holding references keep working and in-place updates are
        visible to both modules.
        """
        if not isinstance(module, nn.RMSNorm):
            raise TypeError(
                f"from_torch expects a torch.nn.RMSNorm, got {type(module).__name__}"
            )
        new = cls(
            tuple(module.normalized_shape),
            eps=module.eps,
            elementwise_affine=False,
        )
        new.elementwise_affine = module.elementwise_affine
        new.weight = module.weight  # registers the Parameter (or None) under the same name
        new.training = module.training
        return new


def replace_rmsnorm(model: nn.Module) -> int:
    """Recursively swap exact-type ``nn.RMSNorm`` submodules for :class:`RMSNorm`.

    Same contract as :func:`fused_layernorm.replace_layernorm`: only modules
    whose exact type is ``torch.nn.RMSNorm`` with a one-dimensional
    ``normalized_shape`` are replaced, parameters are shared (not copied),
    ``model`` itself is never replaced, nothing global is monkeypatched.
    Returns the number of modules replaced.
    """
    return _replace_modules(
        model,
        nn.RMSNorm,
        RMSNorm.from_torch,
        predicate=lambda m: len(m.normalized_shape) == 1,
    )
