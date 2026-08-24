"""Shared helpers for the fused_layernorm test suite.

Everything here is op-agnostic (device/skip logic, tolerances, tensor
factories) or reused by more than one test file (the LayerNorm reference,
which the fused-add tests also need to check ``out == norm(new_residual)``).
Import with ``from _helpers import ...`` — pytest prepends each test file's
directory to ``sys.path``, so this works however the suite is invoked.

``FUSED_LN_TEST_DEVICE`` (default ``"cuda"``) selects the device the GPU-group
tests allocate on; see the note in ``test_layernorm.py``'s module docstring.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

import fused_layernorm

DEVICE = os.environ.get("FUSED_LN_TEST_DEVICE", "cuda")

# The compiled extension module (``None`` when it is not built).
_ext = fused_layernorm.layernorm._ext

_backend_ok = (
    fused_layernorm.is_available() if DEVICE == "cuda" else _ext is not None
)
requires_cuda_ext = pytest.mark.skipif(
    not _backend_ok,
    reason="fused_layernorm_cuda extension or CUDA is unavailable",
)
requires_real_cuda = pytest.mark.skipif(
    DEVICE != "cuda" or not torch.cuda.is_available(),
    reason="needs a real CUDA device (FUSED_LN_TEST_DEVICE is not 'cuda' or CUDA is unavailable)",
)

# Numerical-agreement tolerances vs. the PyTorch reference.  fp16 / bf16
# outputs are compared against references computed in fp32 and cast back
# (PyTorch also accumulates in fp32 for those dtypes); the rtol values for
# those two are torch.testing.assert_close's own dtype defaults.
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-4),
    torch.float64: dict(atol=1e-12, rtol=1e-10),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
    torch.bfloat16: dict(atol=5e-2, rtol=1.6e-2),
}

SHAPES = [
    (1, 1),
    (1, 17),
    (13, 13),
    (17, 1023),
    (32, 768),
    (32, 1024),
    (32, 4096),
    (64, 4096),
    (1, 4095),
    (1, 4097),
    (1, 32768),
    (1000, 1),
    (512, 512),
    (2048, 4096),
]

ALL_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def _randn(shape: Tuple[int, ...], dtype: torch.dtype, device: str = DEVICE) -> torch.Tensor:
    gen_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return torch.randn(shape, dtype=gen_dtype, device=device).to(dtype)


def _affine(
    n: int, dtype: torch.dtype, device: str = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random, deliberately non-identity weight and bias of length ``n``."""
    gen_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    weight = torch.empty(n, dtype=gen_dtype, device=device).uniform_(0.5, 1.5).to(dtype)
    bias = torch.empty(n, dtype=gen_dtype, device=device).normal_().to(dtype)
    return weight, bias


def _ref_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
    approximate: Optional[str] = None,
) -> torch.Tensor:
    """``F.layer_norm`` over the last dim (optionally followed by ``F.gelu``).

    For fp16 / bf16 inputs the whole reference is computed in fp32 and cast
    back once at the end, mirroring what both PyTorch and the kernel do.
    """
    n = x.shape[-1]
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        weight = None if weight is None else weight.float()
        bias = None if bias is None else bias.float()
    y = F.layer_norm(x, (n,), weight, bias, eps)
    if approximate is not None:
        y = F.gelu(y, approximate=approximate)
    return y.to(out_dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, **TOL[expected.dtype])
