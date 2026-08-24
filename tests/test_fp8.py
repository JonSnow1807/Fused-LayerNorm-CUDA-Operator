"""Tests for the fp8-E4M3 quantised-output RMSNorm ops.

The load-bearing check is BYTE EQUALITY against the documented contract:
fp8_out == round_e4m3(clamp(plain_rms_norm_output * (1/scale), ±448)) — the
reciprocal multiply, not a division (they round differently; the contract is
the multiply, matching vLLM's hoisted 1/scale). Everything the plain-norm
suite established then transfers to the quantised ops.
"""

from __future__ import annotations

import pytest
import torch

from fused_layernorm import fused_add_rms_norm_fp8, rms_norm, rms_norm_fp8

from _helpers import DEVICE, _affine, _randn, requires_cuda_ext

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="torch build lacks float8_e4m3fn"
)

FP8_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FP8_SHAPES = [(64, 1024), (17, 1023), (3, 5, 64)]


def _quantize_ref(y: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    q = (y.float() * (1.0 / scale)).clamp(-448.0, 448.0)
    return q.to(torch.float8_e4m3fn)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


# --------------------------------------------------------------------------- #
# CPU fallback (always runs)
# --------------------------------------------------------------------------- #


def test_cpu_fallback_and_grad_rejection() -> None:
    x = torch.randn(4, 64)
    out, s = rms_norm_fp8(x, (64,), None, 1e-6)
    assert out.dtype == torch.float8_e4m3fn and s.shape == (4, 1)
    deq_err = (out.float() * s - torch.nn.functional.rms_norm(x, (64,), None, 1e-6)).abs()
    assert deq_err.max().item() < s.max().item() * 32  # within e4m3 resolution

    xg = torch.randn(4, 64, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        rms_norm_fp8(xg, (64,), None, 1e-6)
    with pytest.raises(ValueError, match="scale_ub"):
        rms_norm_fp8(x, (64,), None, 1e-6, scale=torch.tensor([0.1]), scale_ub=1.0)
    with pytest.raises(ValueError, match="float32/float16/bfloat16"):
        rms_norm_fp8(x.double(), (64,), None, 1e-6)


# --------------------------------------------------------------------------- #
# GPU: byte-exact composite equivalence
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=str)
@pytest.mark.parametrize("shape", FP8_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_dynamic_byte_equals_quantized_plain_norm(shape, dtype) -> None:
    x = _randn(shape, dtype)
    w, _ = _affine(shape[-1], dtype)
    with torch.no_grad():
        out, s = rms_norm_fp8(x, (shape[-1],), w, 1e-6)
        y_plain = rms_norm(x, (shape[-1],), w, 1e-6)
    assert out.dtype == torch.float8_e4m3fn
    assert s.shape == shape[:-1] + (1,) and s.dtype == torch.float32
    # scale is amax/448 of the plain output
    torch.testing.assert_close(
        s, y_plain.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0,
        atol=1e-9, rtol=1e-6,
    )
    assert torch.equal(_bytes(out), _bytes(_quantize_ref(y_plain, s)))


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_static_byte_equals_quantized_plain_norm(dtype) -> None:
    x = _randn((64, 1024), dtype)
    w, _ = _affine(1024, dtype)
    scale = torch.tensor([0.02], device=DEVICE)
    with torch.no_grad():
        out, s_back = rms_norm_fp8(x, (1024,), w, 1e-6, scale=scale)
        y_plain = rms_norm(x, (1024,), w, 1e-6)
    assert s_back is scale
    assert torch.equal(_bytes(out), _bytes(_quantize_ref(y_plain, 0.02)))
    # saturation: values beyond 448*scale clamp to the finite max, never inf/nan
    big = (torch.ones(4, 64, device=DEVICE) * 1000).to(dtype)
    out_b, _ = rms_norm_fp8(big, (64,), None, 1e-6, scale=torch.tensor([1e-4], device=DEVICE))
    assert torch.isfinite(out_b.float()).all()
    assert out_b.float().abs().max().item() <= 448.0


@pytest.mark.cuda
@requires_cuda_ext
def test_fused_add_fp8_dynamic_and_inplace() -> None:
    x = _randn((32, 512), torch.float16)
    r = _randn((32, 512), torch.float16)
    w, _ = _affine(512, torch.float16)
    with torch.no_grad():
        out, z, s = fused_add_rms_norm_fp8(x, r, (512,), w, 1e-6)
        assert torch.equal(z, x + r)  # residual stream stays fp16
        y_plain = rms_norm(z, (512,), w, 1e-6)
        assert torch.equal(_bytes(out), _bytes(_quantize_ref(y_plain, s)))
        # inplace: residual mutated, identical bytes
        r2 = r.clone()
        out_i, z_i, s_i = fused_add_rms_norm_fp8(x, r2, (512,), w, 1e-6, inplace=True)
        assert z_i.data_ptr() == r2.data_ptr()
        assert torch.equal(_bytes(out_i), _bytes(out)) and torch.equal(s_i, s)


@pytest.mark.cuda
@requires_cuda_ext
def test_dynamic_edge_cases() -> None:
    # all-zero rows: finite scale, zero output
    z0 = torch.zeros(4, 64, device=DEVICE, dtype=torch.float16)
    out0, s0 = rms_norm_fp8(z0, (64,), None, 1e-6)
    assert torch.isfinite(s0).all() and (out0.float() == 0).all()
    # scale_ub clamps the amax
    xb = _randn((8, 128), torch.float32) * 100
    _, s_ub = rms_norm_fp8(xb, (128,), None, 1e-6, scale_ub=10.0)
    assert (s_ub <= 10.0 / 448.0 + 1e-9).all()


@pytest.mark.cuda
@requires_cuda_ext
def test_opcheck_and_compile_fp8() -> None:
    x = _randn((8, 256), torch.float16)
    w, _ = _affine(256, torch.float16)
    scale = torch.tensor([0.05], device=DEVICE)
    torch.library.opcheck(
        torch.ops.fused_layernorm.rms_norm_fp8_dynamic.default, (x, w, 1e-6, None)
    )
    torch.library.opcheck(
        torch.ops.fused_layernorm.rms_norm_fp8_static.default, (x, scale, w, 1e-6)
    )

    def fn(x, w):
        out, s = rms_norm_fp8(x, (256,), w, 1e-6)
        return out.float() * s  # dequantised

    with torch.no_grad():
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(fn)(x, w)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(fn, fullgraph=True)
        torch.testing.assert_close(compiled(x, w), fn(x, w))
