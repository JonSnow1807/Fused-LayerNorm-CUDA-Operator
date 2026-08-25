"""Tests for the fp8-E4M3 quantised-output norm ops (RMS and LayerNorm).

The load-bearing check is BYTE EQUALITY against the documented contract:
fp8_out == round_e4m3(clamp(plain_rms_norm_output * (1/scale), ±448)) — the
reciprocal multiply, not a division (they round differently; the contract is
the multiply, matching vLLM's hoisted 1/scale). Everything the plain-norm
suite established then transfers to the quantised ops.
"""

from __future__ import annotations

import pytest
import torch

from fused_layernorm import (
    fused_add_layer_norm_fp8,
    fused_add_rms_norm_fp8,
    layer_norm,
    layer_norm_fp8,
    rms_norm,
    rms_norm_fp8,
)

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


# --------------------------------------------------------------------------- #
# GPU: NaN semantics (kernel must match the eager composite exactly)
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=str)
def test_nan_poisons_values_and_dynamic_scale(dtype) -> None:
    """A NaN input must surface as NaN — never quantise to a finite value.

    Two independent kernel behaviours are on trial: the value path (SATFINITE
    clamp must not turn NaN into ±448) and the dynamic-scale row-amax (fmaxf
    silently DROPS NaN, so a naive reduction gives a NaN row a tiny finite
    scale; torch.amax — the eager composite — propagates it).
    """
    x = _randn((6, 512), dtype)
    x[2, 7] = float("nan")  # poisons row 2's stats, hence the whole row
    w, _ = _affine(512, dtype)
    with torch.no_grad():
        out, s = rms_norm_fp8(x, (512,), w, 1e-6)
        assert torch.isnan(s[2]).all()  # scale propagates, not the 1e-12 floor
        assert torch.isnan(out[2].float()).all()
        # clean rows are untouched by the poisoned one
        y = rms_norm(x, (512,), w, 1e-6)
        clean = [i for i in range(6) if i != 2]
        ref_s = y[clean].float().abs().amax(-1, keepdim=True) / 448.0
        assert ((s[clean] - ref_s).abs() <= ref_s * 1e-6 + 1e-12).all()
        assert torch.equal(_bytes(out[clean]), _bytes(_quantize_ref(y[clean], s[clean])))
        # static mode: value passthrough alone (finite scale, NaN value)
        out_st, _ = rms_norm_fp8(x, (512,), w, 1e-6, scale=torch.tensor([0.02], device=DEVICE))
        assert torch.isnan(out_st[2].float()).all()
        assert torch.isfinite(out_st[clean].float()).all()
        # scale_ub must not un-poison the scale either
        _, s_ub = rms_norm_fp8(x, (512,), w, 1e-6, scale_ub=1.0)
        assert torch.isnan(s_ub[2]).all()


# --------------------------------------------------------------------------- #
# LayerNorm-family fp8 (v0.5.0): the LN mirror of everything above.
# --------------------------------------------------------------------------- #



def test_layer_norm_fp8_cpu_fallback_and_guards() -> None:
    x = torch.randn(4, 64)
    w = torch.rand(64) + 0.5
    b = torch.randn(64)
    out, s = layer_norm_fp8(x, (64,), w, b)
    assert out.dtype == torch.float8_e4m3fn and s.shape == (4, 1)
    ref = torch.nn.functional.layer_norm(x, (64,), w, b, 1e-5)
    assert (out.float() * s - ref).abs().max().item() < s.max().item() * 32

    xg = torch.randn(4, 64, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        layer_norm_fp8(xg, (64,), w, b)
    with pytest.raises(ValueError, match="scale_ub"):
        layer_norm_fp8(x, (64,), w, b, scale=torch.tensor([0.1]), scale_ub=1.0)


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=str)
@pytest.mark.parametrize("shape", FP8_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_ln_dynamic_byte_equals_quantized_plain_norm(shape, dtype) -> None:
    x = _randn(shape, dtype)
    w, b = _affine(shape[-1], dtype)
    with torch.no_grad():
        out, s = layer_norm_fp8(x, (shape[-1],), w, b, 1e-5)
        y = layer_norm(x, (shape[-1],), w, b, 1e-5)
    assert torch.equal(_bytes(out), _bytes(_quantize_ref(y, s)))
    amax = y.float().abs().amax(-1, keepdim=True).clamp(min=1e-12)
    ref_s = amax / 448.0
    assert ((s - ref_s).abs() <= ref_s * 1e-6 + 1e-12).all()


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=str)
def test_ln_static_byte_contract_and_saturation(dtype) -> None:
    x = _randn((32, 512), dtype)
    w, b = _affine(512, dtype)
    scale = torch.tensor([0.02], device=DEVICE)
    with torch.no_grad():
        out, s_back = layer_norm_fp8(x, (512,), w, b, 1e-5, scale=scale)
        y = layer_norm(x, (512,), w, b, 1e-5)
    assert s_back is scale
    assert torch.equal(_bytes(out), _bytes(_quantize_ref(y, 0.02)))
    big = (torch.ones(4, 64, device=DEVICE) * 1000).to(dtype)
    out_b, _ = layer_norm_fp8(big, (64,), None, None, 1e-5,
                              scale=torch.tensor([1e-4], device=DEVICE))
    assert torch.isfinite(out_b.float()).all()
    assert out_b.float().abs().max().item() <= 448.0


@pytest.mark.cuda
@requires_cuda_ext
def test_ln_fused_add_fp8_dynamic_and_inplace() -> None:
    x = _randn((32, 512), torch.float16)
    r = _randn((32, 512), torch.float16)
    w, b = _affine(512, torch.float16)
    with torch.no_grad():
        out, z, s = fused_add_layer_norm_fp8(x, r, (512,), w, b, 1e-5)
        assert torch.equal(z, x + r)  # residual stream stays fp16
        y = layer_norm(z, (512,), w, b, 1e-5)
        assert torch.equal(_bytes(out), _bytes(_quantize_ref(y, s)))
        # inplace mutates residual and returns it as new_residual
        r2 = r.clone()
        out2, z2, s2 = fused_add_layer_norm_fp8(x, r2, (512,), w, b, 1e-5, inplace=True)
        assert z2 is r2 and torch.equal(r2, x + r)
        assert torch.equal(_bytes(out2), _bytes(out)) and torch.equal(s2, s)


@pytest.mark.cuda
@requires_cuda_ext
def test_ln_nan_poisons_values_and_dynamic_scale() -> None:
    x = _randn((6, 512), torch.float16)
    x[2, 7] = float("nan")
    w, b = _affine(512, torch.float16)
    with torch.no_grad():
        out, s = layer_norm_fp8(x, (512,), w, b, 1e-5)
    assert torch.isnan(s[2]).all()
    assert torch.isnan(out[2].float()).all()
    clean = [i for i in range(6) if i != 2]
    assert torch.isfinite(s[clean]).all()


@pytest.mark.cuda
@requires_cuda_ext
def test_ln_fp8_opcheck_and_compile() -> None:
    x = _randn((8, 256), torch.float16)
    w, b = _affine(256, torch.float16)
    scale = torch.tensor([0.05], device=DEVICE)
    torch.library.opcheck(
        torch.ops.fused_layernorm.layer_norm_fp8_dynamic.default, (x, w, b, 1e-5, None)
    )
    torch.library.opcheck(
        torch.ops.fused_layernorm.layer_norm_fp8_static.default, (x, scale, w, b, 1e-5)
    )

    def fn(x, w, b):
        out, s = layer_norm_fp8(x, (256,), w, b, 1e-5)
        return out.float() * s

    with torch.no_grad():
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(fn)(x, w, b)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(fn, fullgraph=True)
        torch.testing.assert_close(compiled(x, w, b), fn(x, w, b))
