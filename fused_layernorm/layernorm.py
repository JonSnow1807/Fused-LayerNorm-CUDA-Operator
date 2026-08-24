"""Python front end for the ``fused_layernorm_cuda`` extension.

This module wraps the forward-only CUDA kernel exposed by the extension
(``fused_layernorm_cuda.layernorm`` / ``fused_layernorm_cuda.layernorm_gelu``)
behind ``F.layer_norm``-shaped functions and an ``nn.LayerNorm`` subclass.

Design rules (see docstrings below for details):

* The extension is imported lazily and optionally.  This module imports fine on
  a machine without CUDA or without the compiled extension; every entry point
  then falls back to plain PyTorch.
* The kernel is forward-only and returns a tensor with no ``grad_fn``.  Whenever
  autograd would need to record the op (grad mode on and some input requires
  grad) we call ``torch.nn.functional.layer_norm`` instead, so training code
  keeps correct gradients.  Silently breaking autograd would be a correctness
  bug, not an optimisation.
* Only the last dimension is normalised by the kernel, i.e. only a 1-D
  ``normalized_shape`` equal to ``input.shape[-1]`` is eligible for the fused
  path.  Every other case (multi-dim ``normalized_shape``, CPU tensors, ...) is
  handled by PyTorch.
* Nothing in ``torch.nn`` is monkeypatched.  Use :func:`replace_layernorm` on a
  specific model instead.

Limitations: no backward pass; the fused path never runs under autograd, under
CUDA autocast, or with ``weight``/``bias`` whose dtype or device differs from
``input`` (those calls go to PyTorch, which then behaves exactly as
``nn.LayerNorm`` would -- including raising, since PyTorch's own CUDA LayerNorm
rejects a weight/bias dtype different from the input outside autocast); the
kernel is only used for CUDA tensors of dtype float32/float64/float16/bfloat16.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (
    _as_shape,
    _autocast_active,
    _eligible,
    _ext,
    _needs_grad,
    _replace_modules,
    _same_dtype_and_device,
    _Shape,
)

__all__ = [
    "is_available",
    "layer_norm",
    "layer_norm_gelu",
    "LayerNorm",
    "replace_layernorm",
]

# The compiled-extension handle (or None) is shared via _common; re-exported
# here because tests and is_available() read ``fused_layernorm.layernorm._ext``.

def is_available() -> bool:
    """Return ``True`` iff the compiled extension imported and CUDA is usable.

    When this returns ``False`` every function in this module still works, but
    runs the plain PyTorch implementation.
    """
    return _ext is not None and torch.cuda.is_available()


def _use_fused(
    input: torch.Tensor,
    normalized_shape: tuple,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> bool:
    """Decide, per call, whether the CUDA kernel can be used.

    All of the following must hold: the extension is importable, ``input`` is
    a CUDA tensor, ``normalized_shape`` is 1-D and equals ``input.shape[-1]``,
    ``weight``/``bias`` (when given) have ``input``'s dtype and device, no
    CUDA autocast region is active, and no gradient is required (the kernel
    is forward-only).

    The dtype/device and autocast conditions exist so that this wrapper never
    changes PyTorch's behaviour: under ``torch.autocast`` PyTorch runs
    ``layer_norm`` in fp32 and returns fp32, whereas the kernel returns
    ``input.dtype``; and outside autocast PyTorch's CUDA LayerNorm itself
    rejects a ``weight``/``bias`` dtype different from ``input``.  Handing
    those calls to PyTorch keeps this module a drop-in replacement (same
    output dtype, same errors) instead of introducing new semantics.
    """
    return _eligible(
        input, normalized_shape, weight, bias, ext_available=_ext is not None
    )


def layer_norm(
    input: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalisation with the same signature as ``F.layer_norm``.

    The fused CUDA kernel is used iff the extension is available, ``input`` is
    on a CUDA device, ``normalized_shape`` is one-dimensional and equal to
    ``input.shape[-1]``, ``weight``/``bias`` (if given) have the same dtype
    and device as ``input``, no CUDA autocast region is active, and autograd
    does not need to record the op (``torch.is_grad_enabled()`` is False or
    none of ``input``/``weight``/``bias`` requires grad).  In every other case
    this is exactly ``torch.nn.functional.layer_norm``.

    Rationale for the grad rule: the kernel is forward-only and returns a
    tensor without ``grad_fn``.  Using it inside a training graph would
    silently detach the output, which is a correctness bug; therefore any call
    that autograd would record goes to PyTorch.

    Rationale for the dtype / autocast rule: under ``torch.autocast`` PyTorch
    runs ``layer_norm`` in fp32 and returns fp32 (the kernel would return
    ``input.dtype``), and outside autocast PyTorch's CUDA kernel rejects a
    ``weight``/``bias`` dtype different from ``input`` (so does this kernel).
    Both kinds of call therefore go to PyTorch, so behaviour -- output dtype
    and errors alike -- is unchanged from ``nn.LayerNorm``.
    """
    shape = _as_shape(normalized_shape)
    # Through the dispatcher (not the raw pybind call) so the same path works
    # under torch.compile without a graph break; see _ops.py. Since v0.4.0
    # gradient-requiring calls no longer fall back to PyTorch: they run the
    # fwd_train op (whose output is bitwise identical to the inference op)
    # with a real CUDA backward attached via register_autograd.
    if _eligible(input, shape, weight, bias, ext_available=_ext is not None,
                 needs_grad_ok=True):
        if _needs_grad(input, weight, bias):
            y, _, _ = torch.ops.fused_layernorm.layer_norm_fwd_train(input, weight, bias, eps)
            return y
        return torch.ops.fused_layernorm.layer_norm(input, weight, bias, eps)
    return F.layer_norm(input, shape, weight, bias, eps)


def layer_norm_gelu(
    input: torch.Tensor,
    normalized_shape: _Shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    approximate: str = "none",
) -> torch.Tensor:
    """``F.gelu(F.layer_norm(...), approximate=approximate)`` in one kernel.

    ``approximate`` is ``"none"`` (erf GELU, the PyTorch default) or
    ``"tanh"``.  The fused kernel is selected under the same rule as
    :func:`layer_norm`; otherwise the two PyTorch ops are applied in sequence.
    Forward-only on the fused path (see :func:`layer_norm`).
    """
    shape = _as_shape(normalized_shape)
    if _use_fused(input, shape, weight, bias):
        return torch.ops.fused_layernorm.layer_norm_gelu(input, weight, bias, eps, approximate)
    return F.gelu(F.layer_norm(input, shape, weight, bias, eps), approximate=approximate)


class LayerNorm(nn.LayerNorm):
    """``torch.nn.LayerNorm`` whose forward routes through :func:`layer_norm`.

    Subclassing keeps ``__init__``, ``reset_parameters``, ``state_dict``
    layout and ``extra_repr`` identical to ``nn.LayerNorm``.  Backend selection
    (fused kernel vs PyTorch) happens per call inside :func:`layer_norm`, so the
    repr deliberately does not claim a backend.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    @classmethod
    def from_torch(cls, module: nn.LayerNorm) -> "LayerNorm":
        """Build a :class:`LayerNorm` that shares parameters with ``module``.

        ``normalized_shape``, ``eps`` and ``elementwise_affine`` are copied and,
        when the module is affine, the *same* ``weight``/``bias`` Parameter
        objects are attached (not clones), so optimisers that already hold
        references to them keep working and in-place updates are visible to
        both modules.
        """
        if not isinstance(module, nn.LayerNorm):
            raise TypeError(
                f"from_torch expects a torch.nn.LayerNorm, got {type(module).__name__}"
            )
        # ``bias`` kwarg exists on nn.LayerNorm since torch 2.1; construct with
        # affine=False and attach the original parameters ourselves so the
        # same code path works on every supported torch version.
        new = cls(
            tuple(module.normalized_shape),
            eps=module.eps,
            elementwise_affine=False,
        )
        new.elementwise_affine = module.elementwise_affine
        # Registering under the same names keeps the state_dict keys identical.
        new.weight = module.weight  # nn.Module.__setattr__ registers Parameters (or None)
        new.bias = module.bias
        new.training = module.training
        return new


def replace_layernorm(model: nn.Module) -> int:
    """Recursively swap ``nn.LayerNorm`` submodules of ``model`` for :class:`LayerNorm`.

    Only modules whose exact type is ``torch.nn.LayerNorm`` (``type(m) is
    nn.LayerNorm``) with a one-dimensional ``normalized_shape`` are replaced;
    custom subclasses and multi-dim LayerNorms are left untouched.  Each
    replacement is created with :meth:`LayerNorm.from_torch`, so parameters are
    shared, not copied.  Works through ``nn.Sequential`` / ``nn.ModuleList`` /
    ``nn.ModuleDict`` because it uses ``named_children`` + ``setattr``.
    ``model`` itself is never replaced (only its descendants).

    Returns the number of modules replaced.

    This function deliberately does *not* monkeypatch ``torch.nn.LayerNorm``
    globally (an earlier version of this package did).  Global patching changes
    the behaviour of every library in the process, including ones that never
    asked for it, and is very hard to undo; a per-model, opt-in replacement is
    the only safe interface.
    """
    return _replace_modules(
        model,
        nn.LayerNorm,
        LayerNorm.from_torch,
        predicate=lambda m: len(m.normalized_shape) == 1,
    )
