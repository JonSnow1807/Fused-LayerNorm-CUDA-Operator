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

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "is_available",
    "layer_norm",
    "layer_norm_gelu",
    "LayerNorm",
    "replace_layernorm",
]

# Lazy / optional import of the compiled extension.  ``_ext`` stays ``None`` when
# the extension has not been built (or cannot be loaded), and every public
# function below then uses the PyTorch fallback.
try:  # pragma: no cover - exercised only when the extension is built
    import fused_layernorm_cuda as _ext  # type: ignore[import-not-found]
except ImportError:
    _ext = None

_Shape = Union[int, Sequence[int], torch.Size]


def is_available() -> bool:
    """Return ``True`` iff the compiled extension imported and CUDA is usable.

    When this returns ``False`` every function in this module still works, but
    runs the plain PyTorch implementation.
    """
    return _ext is not None and torch.cuda.is_available()


def _as_shape(normalized_shape: _Shape) -> tuple:
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)


def _needs_grad(*tensors: Optional[torch.Tensor]) -> bool:
    """True if autograd would need to record an op over ``tensors``."""
    return torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in tensors
    )


def _same_dtype_and_device(input: torch.Tensor, *tensors: Optional[torch.Tensor]) -> bool:
    """True if every given (non-None) tensor has ``input``'s dtype and device."""
    return all(
        t is None or (t.dtype == input.dtype and t.device == input.device) for t in tensors
    )


def _autocast_active() -> bool:
    """True while a CUDA autocast region is active.

    The no-argument form of ``torch.is_autocast_enabled`` reports the CUDA
    autocast state on every torch >= 2.0 (the ``device_type`` argument only
    exists on newer versions); the fused path is CUDA-only, so that is the
    state we need.
    """
    return torch.is_autocast_enabled()


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
    return (
        _ext is not None
        and input.is_cuda
        and input.dim() >= 1
        and len(normalized_shape) == 1
        and normalized_shape[-1] == input.shape[-1]
        and _same_dtype_and_device(input, weight, bias)
        and not _autocast_active()
        and not _needs_grad(input, weight, bias)
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
    if _use_fused(input, shape, weight, bias):
        return _ext.layernorm(input, weight, bias, eps)
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
        return _ext.layernorm_gelu(input, weight, bias, eps, approximate)
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
    count = 0
    for name, child in list(model.named_children()):
        if type(child) is nn.LayerNorm and len(child.normalized_shape) == 1:
            setattr(model, name, LayerNorm.from_torch(child))
            count += 1
        else:
            count += replace_layernorm(child)
    return count
