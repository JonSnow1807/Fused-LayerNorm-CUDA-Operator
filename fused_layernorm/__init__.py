"""fused_layernorm — fused normalisation CUDA kernels for PyTorch.

LayerNorm and RMSNorm with the fusions eager PyTorch lacks: fused
residual-add + norm, fp8-E4M3 quantised outputs, a real CUDA backward, and
torch.compile integration without graph breaks. Public API: the functional
ops (:func:`layer_norm`, :func:`layer_norm_gelu`, :func:`rms_norm`,
:func:`fused_add_layer_norm`, :func:`fused_add_rms_norm`,
:func:`rms_norm_fp8`, :func:`fused_add_rms_norm_fp8`), the drop-in modules
(:class:`LayerNorm`, :class:`RMSNorm`, :class:`FusedAddLayerNorm`,
:class:`FusedAddRMSNorm`), the per-model replacement helpers
(:func:`replace_layernorm`, :func:`replace_rmsnorm` — nothing global is
monkeypatched), and :func:`is_available`.

Every op falls back to the equivalent PyTorch composite when the fused kernel
does not apply; importing this package never requires the compiled extension
or a GPU. See the README for contracts and measured performance.
"""

from .layernorm import (
    LayerNorm,
    is_available,
    layer_norm,
    layer_norm_gelu,
    replace_layernorm,
)
from .rms_norm import (
    RMSNorm,
    replace_rmsnorm,
    rms_norm,
)
from .fused_add import (
    FusedAddLayerNorm,
    FusedAddRMSNorm,
    fused_add_layer_norm,
    fused_add_rms_norm,
)
from .quant import (
    fused_add_rms_norm_fp8,
    rms_norm_fp8,
)

# Registers torch.ops.fused_layernorm.* (needed by the fused paths above and
# by torch.compile tracing). Import order matters only in that this must run
# before any fused-path call; package import guarantees it.
from . import _ops  # noqa: E402,F401

# Single source of truth for the version: setup.py regex-reads this line and
# injects it into the extension as -DFUSED_LN_VERSION; pyproject.toml reads it
# via [tool.setuptools.dynamic].
__version__ = "0.4.0"

__all__ = [
    "FusedAddLayerNorm",
    "FusedAddRMSNorm",
    "LayerNorm",
    "RMSNorm",
    "fused_add_layer_norm",
    "fused_add_rms_norm",
    "fused_add_rms_norm_fp8",
    "is_available",
    "layer_norm",
    "layer_norm_gelu",
    "replace_layernorm",
    "replace_rmsnorm",
    "rms_norm",
    "rms_norm_fp8",
    "__version__",
]
