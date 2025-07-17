"""
Fused LayerNorm CUDA Package

High-performance drop-in replacement for torch.nn.LayerNorm with CUDA acceleration.
"""

from .layernorm import FusedLayerNorm, fused_layer_norm, replace_torch_layernorm, restore_torch_layernorm
from .functional import FusedLayerNormFunction

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "FusedLayerNorm",
    "FusedLayerNormFunction", 
    "fused_layer_norm",
    "replace_torch_layernorm",
    "restore_torch_layernorm",
]

# Check CUDA availability
import torch
if not torch.cuda.is_available():
    import warnings
    warnings.warn(
        "CUDA is not available. FusedLayerNorm will fall back to PyTorch's native implementation.",
        RuntimeWarning
    )