"""Tests for rms_norm / RMSNorm / replace_rmsnorm.

The reference is ``F.rms_norm`` with an EXPLICIT eps (never ``None`` — PyTorch
substitutes the machine epsilon of the computation dtype for ``None``, which
is its own dedicated test here). fp16/bf16 are compared against the fp32
composite, mirroring the LayerNorm tests.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_layernorm
from fused_layernorm import RMSNorm, replace_rmsnorm, rms_norm

from _helpers import (
    ALL_DTYPES,
    DEVICE,
    SHAPES,
    TOL,
    _affine,
    _assert_close,
    _randn,
    requires_cuda_ext,
)


def _ref_rms_norm(
    x: torch.Tensor, weight: Optional[torch.Tensor], eps: Optional[float] = 1e-6
) -> torch.Tensor:
    """``F.rms_norm`` computed in fp32 for half dtypes, cast back once."""
    n = x.shape[-1]
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        weight = None if weight is None else weight.float()
    y = F.rms_norm(x, (n,), weight, eps)
    return y.to(out_dtype)


# --------------------------------------------------------------------------- #
# CPU fallback (always runs)
# --------------------------------------------------------------------------- #


def test_cpu_fallback_matches_torch() -> None:
    x = torch.randn(4, 7, 64)
    w = torch.rand(64) + 0.5
    torch.testing.assert_close(rms_norm(x, (64,), w, 1e-6), F.rms_norm(x, (64,), w, 1e-6))
    # eps=None must keep torch's machine-epsilon semantics on the fallback
    torch.testing.assert_close(rms_norm(x, (64,), w, None), F.rms_norm(x, (64,), w, None))
    # multi-dim normalized_shape goes to torch verbatim
    torch.testing.assert_close(
        rms_norm(x, (7, 64), None, 1e-6), F.rms_norm(x, (7, 64), None, 1e-6)
    )


def test_module_cpu_matches_nn_rmsnorm() -> None:
    m_ref = nn.RMSNorm(96)
    m = RMSNorm.from_torch(m_ref)
    x = torch.randn(5, 96)
    torch.testing.assert_close(m(x), m_ref(x))
    assert list(m.state_dict().keys()) == list(m_ref.state_dict().keys())
    assert m.weight is m_ref.weight  # shared, not copied


# --------------------------------------------------------------------------- #
# GPU tests
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_kernel_matches_rms_norm(shape, dtype: torch.dtype) -> None:
    x = _randn(shape, dtype)
    w, _ = _affine(shape[-1], dtype)
    y = rms_norm(x, (shape[-1],), w, 1e-6)
    _assert_close(y, _ref_rms_norm(x, w, 1e-6))


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_kernel_weightless(dtype: torch.dtype) -> None:
    x = _randn((64, 512), dtype)
    _assert_close(rms_norm(x, (512,), None, 1e-6), _ref_rms_norm(x, None, 1e-6))


@pytest.mark.cuda
@requires_cuda_ext
def test_eps_none_matches_torch_machine_eps_semantics() -> None:
    """The fused path must resolve eps=None to torch's exact value.

    F.rms_norm(eps=None) uses torch.finfo(compute_dtype).eps — NOT 1e-5/1e-6.
    A drop-in replacement that substituted a conventional eps would differ
    measurably on small-magnitude rows.
    """
    for dtype in (torch.float32, torch.float16, torch.float64):
        x = _randn((32, 256), dtype) * 1e-3  # small values make eps visible
        got = rms_norm(x, (256,), None, None)
        want = F.rms_norm(
            x.float() if dtype == torch.float16 else x, (256,), None, None
        ).to(dtype)
        torch.testing.assert_close(got, want, **TOL[dtype])


@pytest.mark.cuda
@requires_cuda_ext
def test_inference_no_gradfn_and_training_grads() -> None:
    x = _randn((8, 128), torch.float32)
    w, _ = _affine(128, torch.float32)
    with torch.no_grad():
        assert rms_norm(x, (128,), w, 1e-6).grad_fn is None
    xg = x.clone().requires_grad_()
    wg = w.clone().requires_grad_()
    y = rms_norm(xg, (128,), wg, 1e-6)  # fused fwd-train path: grads work
    assert y.grad_fn is not None
    y.sum().backward()
    assert xg.grad is not None and wg.grad is not None


@pytest.mark.cuda
@requires_cuda_ext
def test_non_contiguous_and_storage_offset() -> None:
    base = torch.randn(512 * 1024 + 1, device=DEVICE)
    x = base[1:].view(512, 1024)  # contiguous but 16-byte-misaligned
    assert x.data_ptr() % 16 != 0
    _assert_close(rms_norm(x, (1024,), None, 1e-6), _ref_rms_norm(x, None, 1e-6))
    xt = torch.randn(64, 32, device=DEVICE).t()  # non-contiguous
    _assert_close(rms_norm(xt, (64,), None, 1e-6), _ref_rms_norm(xt.contiguous(), None, 1e-6))


@pytest.mark.cuda
@requires_cuda_ext
def test_module_gpu_and_replace() -> None:
    inner = nn.Sequential(nn.Linear(64, 64), nn.RMSNorm(64))
    model = nn.Sequential(inner, nn.RMSNorm(64), nn.LayerNorm(64)).to(DEVICE)
    n = replace_rmsnorm(model)
    assert n == 2
    assert type(model[1]) is RMSNorm and type(inner[1]) is RMSNorm
    assert type(model[2]) is nn.LayerNorm  # untouched
    x = torch.randn(4, 64, device=DEVICE)
    ref = nn.RMSNorm(64).to(DEVICE)
    with torch.no_grad():
        ref.weight.copy_(model[1].weight)
        torch.testing.assert_close(model[1](x), ref(x), **TOL[torch.float32])


@pytest.mark.cuda
@requires_cuda_ext
def test_rms_norm_determinism() -> None:
    x = _randn((256, 1024), torch.float32)
    w, _ = _affine(1024, torch.float32)
    a = rms_norm(x, (1024,), w, 1e-6)
    b = rms_norm(x, (1024,), w, 1e-6)
    assert torch.equal(a, b)


@pytest.mark.cuda
@requires_cuda_ext
def test_opcheck_rms_norm() -> None:
    x = _randn((8, 256), torch.float32)
    w, _ = _affine(256, torch.float32)
    torch.library.opcheck(torch.ops.fused_layernorm.rms_norm.default, (x, w, 1e-6))
    torch.library.opcheck(torch.ops.fused_layernorm.rms_norm.default, (x, None, 1e-6))


@pytest.mark.cuda
@requires_cuda_ext
def test_rms_norm_compiles_fullgraph() -> None:
    x = _randn((64, 1024), torch.float32)
    w, _ = _affine(1024, torch.float32)

    def fn(x, w):
        return fused_layernorm.rms_norm(x, (1024,), w, 1e-6)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(x, w)
    assert explanation.graph_break_count == 0
    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(x, w), fn(x, w))
