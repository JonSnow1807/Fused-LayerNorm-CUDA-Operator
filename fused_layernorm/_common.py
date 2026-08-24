"""Op-agnostic helpers shared by every fused_layernorm op wrapper.

These were originally private helpers of ``layernorm.py``; they are hoisted
here unchanged so the RMSNorm / fused-add / quant wrappers can reuse them.
Nothing in this module imports the compiled extension.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

_Shape = Union[int, Sequence[int], torch.Size]


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
    exists on newer versions); the fused paths are CUDA-only, so that is the
    state we need.
    """
    return torch.is_autocast_enabled()


def _eligible(
    input: torch.Tensor,
    normalized_shape: tuple,
    *tensors: Optional[torch.Tensor],
    ext_available: bool,
    needs_grad_ok: bool = False,
) -> bool:
    """Decide, per call, whether a fused CUDA op can be used.

    Op-agnostic generalization of the 0.3.0 ``_use_fused`` rule: the extension
    must be importable, ``input`` must be a CUDA tensor normalised over a 1-D
    ``normalized_shape`` equal to its last dimension, every extra tensor
    (affine params, residual, ...) must share ``input``'s dtype and device, no
    CUDA autocast region may be active, and — unless the op has a registered
    backward (``needs_grad_ok=True``) — autograd must not need to record the
    op.  The rationale for each condition is documented in
    ``layernorm.layer_norm``'s docstring.
    """
    return (
        ext_available
        and input.is_cuda
        and input.dim() >= 1
        and len(normalized_shape) == 1
        and normalized_shape[-1] == input.shape[-1]
        and _same_dtype_and_device(input, *tensors)
        and not _autocast_active()
        and (needs_grad_ok or not _needs_grad(input, *tensors))
    )


def _resolve_rms_eps(dtype: torch.dtype, eps: Optional[float]) -> float:
    """Replicate ``F.rms_norm``'s ``eps=None`` semantics.

    PyTorch substitutes the machine epsilon of the *computation* dtype: fp32
    accumulation for fp16/bf16/fp32 inputs, fp64 for fp64 inputs (see
    torch/_decomp/decompositions.py, rms_norm decomposition).  This is NOT
    1e-5/1e-6; a drop-in replacement must match it exactly.
    """
    if eps is not None:
        return eps
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return torch.finfo(acc_dtype).eps


def _replace_modules(
    model: nn.Module,
    source_type: Type[nn.Module],
    factory: Callable[[nn.Module], nn.Module],
    predicate: Callable[[nn.Module], bool] = lambda m: True,
) -> int:
    """Recursively swap exact-type ``source_type`` children of ``model``.

    Only modules whose exact type is ``source_type`` (``type(m) is
    source_type``) are replaced, so custom subclasses are left untouched.
    Works through ``nn.Sequential`` / ``nn.ModuleList`` / ``nn.ModuleDict``
    because it uses ``named_children`` + ``setattr``.  ``model`` itself is
    never replaced (only its descendants).  Returns the number of modules
    replaced.  Deliberately does not monkeypatch anything global — see
    ``layernorm.replace_layernorm``'s docstring for the rationale.
    """
    count = 0
    for name, child in list(model.named_children()):
        if type(child) is source_type and predicate(child):
            setattr(model, name, factory(child))
            count += 1
        else:
            count += _replace_modules(child, source_type, factory, predicate)
    return count
