"""Tests for fused_add_layer_norm / fused_add_rms_norm and their modules.

The load-bearing checks are STRUCTURAL AND BITWISE, not tolerance-based:
  * ``new_residual == input + residual`` exactly (a single eager add rounds
    identically to the kernel's add-then-round), and
  * ``out == plain_norm(new_residual)`` exactly (the kernel computes its
    statistics over the rounded sum, so the fused op is bit-for-bit the
    unfused composite).
Everything the tolerance-based suites already established about the plain
norms then transfers to the fused ops for free.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import fused_layernorm
from fused_layernorm import (
    FusedAddLayerNorm,
    FusedAddRMSNorm,
    fused_add_layer_norm,
    fused_add_rms_norm,
    layer_norm,
    rms_norm,
)

from _helpers import ALL_DTYPES, DEVICE, _affine, _randn, requires_cuda_ext

FUSED_SHAPES = [(1, 1), (17, 1023), (64, 1024), (3, 5, 64), (512, 512), (2048, 4096)]


def _inputs(shape, dtype):
    x = _randn(shape, dtype)
    r = _randn(shape, dtype)
    w, b = _affine(shape[-1], dtype)
    return x, r, w, b


# --------------------------------------------------------------------------- #
# CPU fallback (always runs)
# --------------------------------------------------------------------------- #


def test_cpu_fallback_composite() -> None:
    x, r = torch.randn(4, 64), torch.randn(4, 64)
    w = torch.rand(64) + 0.5
    b = torch.randn(64)
    out, z = fused_add_layer_norm(x, r, (64,), w, b, 1e-5)
    torch.testing.assert_close(z, x + r)
    torch.testing.assert_close(out, F.layer_norm(x + r, (64,), w, b, 1e-5))
    out2, z2 = fused_add_rms_norm(x, r, (64,), w, None)
    torch.testing.assert_close(z2, x + r)
    torch.testing.assert_close(out2, F.rms_norm(x + r, (64,), w, None))


def test_inplace_rejected_under_grad_everywhere() -> None:
    x = torch.randn(4, 64, requires_grad=True)
    r = torch.randn(4, 64)
    with pytest.raises(RuntimeError, match="inference-only"):
        fused_add_layer_norm(x, r, (64,), inplace=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        fused_add_rms_norm(x, r, (64,), inplace=True)
    with torch.no_grad():  # requires_grad flags are ignored when grad mode is off
        fused_add_layer_norm(x, r, (64,), inplace=True)


# --------------------------------------------------------------------------- #
# GPU: bitwise structural contract
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("shape", FUSED_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_bitwise_composite_equivalence(shape, dtype: torch.dtype) -> None:
    x, r, w, b = _inputs(shape, dtype)
    n = shape[-1]

    out, z = fused_add_layer_norm(x, r, (n,), w, b, 1e-5)
    assert torch.equal(z, x + r)
    assert torch.equal(out, layer_norm(z, (n,), w, b, 1e-5))

    out2, z2 = fused_add_rms_norm(x, r, (n,), w, 1e-6)
    assert torch.equal(z2, x + r)
    assert torch.equal(out2, rms_norm(z2, (n,), w, 1e-6))


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_out_of_place_leaves_inputs_untouched(dtype: torch.dtype) -> None:
    x, r, w, b = _inputs((64, 1024), dtype)
    x0, r0 = x.clone(), r.clone()
    out, z = fused_add_layer_norm(x, r, (1024,), w, b, 1e-5)
    assert torch.equal(x, x0) and torch.equal(r, r0)
    assert z.data_ptr() != r.data_ptr() and out.data_ptr() != x.data_ptr()


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("kind", ["ln", "rms"])
def test_inplace_aliases_residual(kind: str) -> None:
    x, r, w, b = _inputs((64, 1024), torch.float16)
    expected_z = x + r
    with torch.inference_mode():
        if kind == "ln":
            out, z = fused_add_layer_norm(x, r, (1024,), w, b, 1e-5, inplace=True)
            ref = layer_norm(expected_z, (1024,), w, b, 1e-5)
        else:
            out, z = fused_add_rms_norm(x, r, (1024,), w, 1e-6, inplace=True)
            ref = rms_norm(expected_z, (1024,), w, 1e-6)
    assert z.data_ptr() == r.data_ptr()  # the residual WAS mutated
    assert torch.equal(z, expected_z)
    assert torch.equal(out, ref)


@pytest.mark.cuda
@requires_cuda_ext
def test_inplace_noncontiguous_residual_rejected() -> None:
    x = _randn((64, 64), torch.float32)
    r = _randn((64, 64), torch.float32).t()  # non-contiguous
    with torch.no_grad(), pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.fused_layernorm.fused_add_layer_norm_(x, r, None, None, 1e-5)


@pytest.mark.cuda
@requires_cuda_ext
def test_grad_calls_use_kernel_and_match_composite() -> None:
    """Autograd calls run the fused fwd_train + CUDA backward (since v0.4.0);
    grads for BOTH outputs must match the eager composite's."""
    x, r, w, b = _inputs((16, 128), torch.float32)
    xg, rg = x.clone().requires_grad_(), r.clone().requires_grad_()
    out, z = fused_add_layer_norm(xg, rg, (128,), w, b, 1e-5)
    assert out.grad_fn is not None and z.grad_fn is not None
    (out.mean() + z.mean()).backward()
    assert xg.grad is not None and rg.grad is not None
    # composite reference grads
    xc, rc = x.clone().requires_grad_(), r.clone().requires_grad_()
    zc = xc + rc
    outc = F.layer_norm(zc, (128,), w, b, 1e-5)
    (outc.mean() + zc.mean()).backward()
    torch.testing.assert_close(xg.grad, xc.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(rg.grad, rc.grad, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# GPU: modules, opcheck, compile
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
def test_fused_add_modules() -> None:
    m = FusedAddRMSNorm(256).to(DEVICE)
    x = _randn((8, 256), torch.float32)
    r = _randn((8, 256), torch.float32)
    with torch.no_grad():
        out, z = m(x, r)
        assert torch.equal(z, x + r)
        assert torch.equal(out, rms_norm(z, (256,), m.weight, m.eps))
        out0, z0 = m(x)  # residual=None: x becomes the residual stream
        assert z0 is x
        torch.testing.assert_close(out0, rms_norm(x, (256,), m.weight, m.eps))

    ln = FusedAddLayerNorm(256, inplace=True).to(DEVICE)
    xg = x.clone().requires_grad_()
    out, z = ln(xg, r.clone())  # grad mode on: inplace module falls back to out-of-place
    assert out.grad_fn is not None
    with torch.no_grad():
        r2 = r.clone()
        out, z = ln(x, r2)
        assert z.data_ptr() == r2.data_ptr()  # inference: genuinely in place


@pytest.mark.cuda
@requires_cuda_ext
def test_opcheck_fused_add_ops() -> None:
    x, r, w, b = _inputs((8, 256), torch.float32)
    torch.library.opcheck(
        torch.ops.fused_layernorm.fused_add_layer_norm.default, (x, r, w, b, 1e-5)
    )
    torch.library.opcheck(
        torch.ops.fused_layernorm.fused_add_rms_norm.default, (x, r, w, 1e-6)
    )
    torch.library.opcheck(
        torch.ops.fused_layernorm.fused_add_layer_norm_.default,
        (x, r.clone(), w, b, 1e-5),
    )
    torch.library.opcheck(
        torch.ops.fused_layernorm.fused_add_rms_norm_.default, (x, r.clone(), w, 1e-6)
    )


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("inplace", [False, True])
def test_fused_add_compiles_fullgraph(inplace: bool) -> None:
    x, r, w, _ = _inputs((64, 1024), torch.float32)

    def fn(x, r, w):
        out, z = fused_add_rms_norm(x, r, (1024,), w, 1e-6, inplace=inplace)
        return out, z

    with torch.no_grad():
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(fn)(x, r.clone(), w)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(fn, fullgraph=True)
        out_c, z_c = compiled(x, r.clone(), w)
        out_e, z_e = fn(x, r.clone(), w)
        assert torch.equal(out_c, out_e) and torch.equal(z_c, z_e)
