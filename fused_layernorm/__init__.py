"""fused_layernorm — a small forward-only fused LayerNorm (+GELU) CUDA kernel for PyTorch.

Public API:

* :func:`layer_norm` / :func:`layer_norm_gelu` — ``F.layer_norm``-shaped
  functions that use the CUDA kernel when it is available and applicable and
  fall back to PyTorch otherwise (always when gradients are required; the kernel
  has no backward pass).
* :class:`LayerNorm` — ``torch.nn.LayerNorm`` subclass routing through
  :func:`layer_norm`; :meth:`LayerNorm.from_torch` wraps an existing module
  sharing its parameters.
* :func:`replace_layernorm` — opt-in, per-model replacement of exact
  ``nn.LayerNorm`` submodules.  Nothing global is monkeypatched.
* :func:`is_available` — whether the compiled extension and CUDA are usable.

Importing this package never requires the compiled extension or a GPU.
"""

from .layernorm import (
    LayerNorm,
    is_available,
    layer_norm,
    layer_norm_gelu,
    replace_layernorm,
)

# Registers torch.ops.fused_layernorm.* (needed by the fused paths above and
# by torch.compile tracing). Import order matters only in that this must run
# before any fused-path call; package import guarantees it.
from . import _ops  # noqa: E402,F401

# Single source of truth for the version: setup.py regex-reads this line and
# injects it into the extension as -DFUSED_LN_VERSION; pyproject.toml reads it
# via [tool.setuptools.dynamic].
__version__ = "0.4.0.dev0"

__all__ = [
    "LayerNorm",
    "is_available",
    "layer_norm",
    "layer_norm_gelu",
    "replace_layernorm",
    "__version__",
]
