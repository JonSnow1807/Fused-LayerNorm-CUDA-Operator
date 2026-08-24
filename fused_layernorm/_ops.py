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
