"""Fused residual-add + normalisation — the op pair PyTorch eager lacks.

Pre-norm transformer blocks compute ``residual = x + residual`` followed by a
norm of the sum, twice per layer.  Eager PyTorch runs that as two kernels and
pays a full extra HBM round-trip of the hidden state; these ops do it in one
kernel.

Contract (both ops): ``new_residual = round(input + residual)`` — rounded to
the input dtype exactly once — and ``out = norm(new_residual)`` with the
statistics computed over the ROUNDED sum, so ``out`` equals a plain
``layer_norm``/``rms_norm`` of ``new_residual`` **bitwise** (the fused op is
indistinguishable from the unfused composite).  Returns
``(out, new_residual)``.

``inplace=True`` (inference-only) writes the sum into ``residual``'s storage
instead of allocating: the returned ``new_residual`` *is* ``residual``, and
the pre-call values are gone.  Calls that require gradients raise — use the
default out-of-place form under autograd (its backward lands with the
training phase; until then autograd-requiring calls fall back to the eager
composite, which is numerically identical).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (
    _as_shape,
    _eligible,
    _ext,
    _needs_grad,
    _resolve_rms_eps,
    _Shape,
)

__all__ = [
    "fused_add_layer_norm",
    "fused_add_rms_norm",
    "FusedAddLayerNorm",
    "FusedAddRMSNorm",
]


def _check_inplace_grad(*tensors: Optional[torch.Tensor]) -> None:
    if _needs_grad(*tensors):
        raise RuntimeError(
            "inplace fused_add is inference-only; use inplace=False under autograd"
        )


def fused_add_layer_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``new_residual = input + residual; out = layer_norm(new_residual)``.

    Returns ``(out, new_residual)``.  Fused-kernel eligibility follows the
    same rule as :func:`fused_layernorm.layer_norm` with ``residual`` held to
    the same dtype/device requirements; ineligible calls run the eager
    composite (numerically identical: a single eager add rounds exactly like
    the kernel's add-then-round).
    """
    shape = _as_shape(normalized_shape)
    if inplace:
        _check_inplace_grad(input, residual, weight, bias)
        if _eligible(input, shape, residual, weight, bias, ext_available=_ext is not None):
            out = torch.ops.fused_layernorm.fused_add_layer_norm_(
                input, residual, weight, bias, eps
            )
            return out, residual
        residual = residual.add_(input)
        return F.layer_norm(residual, shape, weight, bias, eps), residual
    if _eligible(input, shape, residual, weight, bias, ext_available=_ext is not None,
                 needs_grad_ok=True):
        if _needs_grad(input, residual, weight, bias):
            y, z, _, _ = torch.ops.fused_layernorm.fused_add_layer_norm_fwd_train(
                input, residual, weight, bias, eps
            )
            return y, z
        return torch.ops.fused_layernorm.fused_add_layer_norm(
            input, residual, weight, bias, eps
        )
    z = input + residual
    return F.layer_norm(z, shape, weight, bias, eps), z


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    *,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``new_residual = input + residual; out = rms_norm(new_residual)``.

    Returns ``(out, new_residual)``.  ``eps=None`` follows ``F.rms_norm``'s
    machine-epsilon convention (see :func:`fused_layernorm.rms_norm`).
    """
    shape = _as_shape(normalized_shape)
    if inplace:
        _check_inplace_grad(input, residual, weight)
        if _eligible(input, shape, residual, weight, ext_available=_ext is not None):
            out = torch.ops.fused_layernorm.fused_add_rms_norm_(
                input, residual, weight, _resolve_rms_eps(input.dtype, eps)
            )
            return out, residual
        residual = residual.add_(input)
        return F.rms_norm(residual, shape, weight, eps), residual
    if _eligible(input, shape, residual, weight, ext_available=_ext is not None,
                 needs_grad_ok=True):
        eps_c = _resolve_rms_eps(input.dtype, eps)
        if _needs_grad(input, residual, weight):
            y, z, _ = torch.ops.fused_layernorm.fused_add_rms_norm_fwd_train(
                input, residual, weight, eps_c
            )
            return y, z
        return torch.ops.fused_layernorm.fused_add_rms_norm(input, residual, weight, eps_c)
    z = input + residual
    return F.rms_norm(z, shape, weight, eps), z


class _FusedAddNormBase(nn.Module):
    """Shared shape/state plumbing for the two fused-add modules."""

    def __init__(
        self,
        normalized_shape: _Shape,
        eps: Optional[float],
        elementwise_affine: bool,
        bias: bool,
        inplace: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.normalized_shape = _as_shape(normalized_shape)
        if len(self.normalized_shape) != 1:
            raise ValueError(
                "fused-add norms normalise over the last dimension only; "
                f"got normalized_shape={self.normalized_shape}"
            )
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.inplace = inplace
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if getattr(self, "bias", None) is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.normalized_shape)}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, inplace={self.inplace}"
        )

    def _inplace_now(self) -> bool:
        # The inplace kernel is inference-only; under grad mode fall back to
        # the differentiable out-of-place form instead of raising, so the
        # same module works in training and inference loops.
        return self.inplace and not torch.is_grad_enabled()


class FusedAddLayerNorm(_FusedAddNormBase):
    """Pre-norm block module: ``forward(x, residual) -> (normed, new_residual)``.

    With ``residual=None`` the input itself becomes the residual stream:
    returns ``(layer_norm(x), x)``.  Always returns a tuple.  ``inplace=True``
    applies only when gradients are off (see :func:`fused_add_layer_norm`).
    """

    def __init__(
        self,
        normalized_shape: _Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, inplace, device, dtype)

    def forward(
        self, input: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from .layernorm import layer_norm

        if residual is None:
            return (
                layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps),
                input,
            )
        return fused_add_layer_norm(
            input,
            residual,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            inplace=self._inplace_now(),
        )


class FusedAddRMSNorm(_FusedAddNormBase):
    """Pre-norm block module: ``forward(x, residual) -> (normed, new_residual)``.

    RMSNorm flavour (no bias). ``eps=None`` follows ``nn.RMSNorm``'s
    machine-epsilon convention.
    """

    def __init__(
        self,
        normalized_shape: _Shape,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            normalized_shape, eps, elementwise_affine, bias=False, inplace=inplace,
            device=device, dtype=dtype,
        )

    def forward(
        self, input: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from .rms_norm import rms_norm

        if residual is None:
            return rms_norm(input, self.normalized_shape, self.weight, self.eps), input
        return fused_add_rms_norm(
            input,
            residual,
            self.normalized_shape,
            self.weight,
            self.eps,
            inplace=self._inplace_now(),
        )
