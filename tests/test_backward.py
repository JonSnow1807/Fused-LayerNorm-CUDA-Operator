"""Backward-pass tests: gradcheck (fp64), composite-grad comparison, both-output
grads for fused-add, and bitwise determinism of the parameter gradients.

torch.autograd.gradcheck is the strongest tool available and needs fp64 - the
reason the backward kernels support double at all. The fp32/fp16/bf16 checks
compare against the eager composite's grads with per-dtype tolerances.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fused_layernorm import fused_add_layer_norm, fused_add_rms_norm, layer_norm, rms_norm

from _helpers import _affine, _randn, requires_cuda_ext

GRAD_TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-4),
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=5e-2),
}


def _grad_inputs(shape, dtype, *, residual: bool):
    x = _randn(shape, dtype).requires_grad_()
    w, b = _affine(shape[-1], dtype)
    w.requires_grad_()
    b.requires_grad_()
    r = _randn(shape, dtype).requires_grad_() if residual else None
    return x, r, w, b


# --------------------------------------------------------------------------- #
# gradcheck (fp64, CUDA): the kernels must satisfy numeric differentiation.
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("shape", [(3, 17), (2, 5, 32)], ids=lambda s: "x".join(map(str, s)))
def test_gradcheck_layer_norm(shape) -> None:
    x, _, w, b = _grad_inputs(shape, torch.float64, residual=False)
    assert torch.autograd.gradcheck(
        lambda x, w, b: layer_norm(x, (shape[-1],), w, b, 1e-5), (x, w, b)
    )


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("with_weight", [True, False])
def test_gradcheck_rms_norm(with_weight: bool) -> None:
    x, _, w, _ = _grad_inputs((3, 17), torch.float64, residual=False)
    w = w if with_weight else None
    args = (x, w) if with_weight else (x,)
    fn = (lambda x, w: rms_norm(x, (17,), w, 1e-6)) if with_weight else (
        lambda x: rms_norm(x, (17,), None, 1e-6)
    )
    assert torch.autograd.gradcheck(fn, args)


@pytest.mark.cuda
@requires_cuda_ext
def test_gradcheck_fused_add_layer_norm() -> None:
    x, r, w, b = _grad_inputs((3, 17), torch.float64, residual=True)

    def fn(x, r, w, b):
        out, z = fused_add_layer_norm(x, r, (17,), w, b, 1e-5)
        return out, z  # gradcheck exercises BOTH outputs

    assert torch.autograd.gradcheck(fn, (x, r, w, b))


@pytest.mark.cuda
@requires_cuda_ext
def test_gradcheck_fused_add_rms_norm() -> None:
    x, r, w, _ = _grad_inputs((3, 17), torch.float64, residual=True)

    def fn(x, r, w):
        out, z = fused_add_rms_norm(x, r, (17,), w, 1e-6)
        return out, z

    assert torch.autograd.gradcheck(fn, (x, r, w))


# --------------------------------------------------------------------------- #
# Composite-grad comparison across working dtypes.
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("shape", [(17, 1023), (64, 1024), (512, 512)],
                         ids=lambda s: "x".join(map(str, s)))
def test_layer_norm_grads_match_composite(shape, dtype) -> None:
    x, _, w, b = _grad_inputs(shape, dtype, residual=False)
    y = layer_norm(x, (shape[-1],), w, b, 1e-5)
    assert y.grad_fn is not None  # the kernel path, not a fallback
    gy = torch.randn_like(y)
    dx, dw, db = torch.autograd.grad(y, (x, w, b), gy)

    xc = x.detach().clone().requires_grad_()
    wc = w.detach().clone().requires_grad_()
    bc = b.detach().clone().requires_grad_()
    if dtype in (torch.float16, torch.bfloat16):
        ref = F.layer_norm(xc.float(), (shape[-1],), wc.float(), bc.float(), 1e-5)
        dxr, dwr, dbr = torch.autograd.grad(ref, (xc, wc, bc), gy.float())
    else:
        ref = F.layer_norm(xc, (shape[-1],), wc, bc, 1e-5)
        dxr, dwr, dbr = torch.autograd.grad(ref, (xc, wc, bc), gy)
    tol = GRAD_TOL[dtype]
    torch.testing.assert_close(dx, dxr.to(dtype), **tol)
    torch.testing.assert_close(dw, dwr.to(dtype), **tol)
    torch.testing.assert_close(db, dbr.to(dtype), **tol)


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_rms_norm_grads_match_composite(dtype) -> None:
    x, _, w, _ = _grad_inputs((64, 1024), dtype, residual=False)
    y = rms_norm(x, (1024,), w, 1e-6)
    assert y.grad_fn is not None
    gy = torch.randn_like(y)
    dx, dw = torch.autograd.grad(y, (x, w), gy)

    xc = x.detach().clone().requires_grad_()
    wc = w.detach().clone().requires_grad_()
    if dtype == torch.float16:
        ref = F.rms_norm(xc.float(), (1024,), wc.float(), 1e-6)
        dxr, dwr = torch.autograd.grad(ref, (xc, wc), gy.float())
    else:
        ref = F.rms_norm(xc, (1024,), wc, 1e-6)
        dxr, dwr = torch.autograd.grad(ref, (xc, wc), gy)
    tol = GRAD_TOL[dtype]
    torch.testing.assert_close(dx, dxr.to(dtype), **tol)
    torch.testing.assert_close(dw, dwr.to(dtype), **tol)


@pytest.mark.cuda
@requires_cuda_ext
def test_fused_add_both_output_grads() -> None:
    """The classic fused-add bug: forgetting that new_residual carries
    gradient. Feed DIFFERENT cotangents to both outputs and compare every
    input grad against the composite."""
    x, r, w, b = _grad_inputs((32, 256), torch.float32, residual=True)
    out, z = fused_add_layer_norm(x, r, (256,), w, b, 1e-5)
    gy, gz = torch.randn_like(out), torch.randn_like(z)
    dx, dr, dw, db = torch.autograd.grad((out, z), (x, r, w, b), (gy, gz))

    xc = x.detach().clone().requires_grad_()
    rc = r.detach().clone().requires_grad_()
    wc = w.detach().clone().requires_grad_()
    bc = b.detach().clone().requires_grad_()
    zc = xc + rc
    outc = F.layer_norm(zc, (256,), wc, bc, 1e-5)
    dxr, drr, dwr, dbr = torch.autograd.grad((outc, zc), (xc, rc, wc, bc), (gy, gz))

    tol = GRAD_TOL[torch.float32]
    torch.testing.assert_close(dx, dxr, **tol)
    torch.testing.assert_close(dr, drr, **tol)
    torch.testing.assert_close(dw, dwr, **tol)
    torch.testing.assert_close(db, dbr, **tol)
    assert torch.equal(dx, dr)  # z = x + r: identical grads by construction


@pytest.mark.cuda
@requires_cuda_ext
def test_fused_add_rms_both_output_grads() -> None:
    x, r, w, _ = _grad_inputs((32, 256), torch.float32, residual=True)
    out, z = fused_add_rms_norm(x, r, (256,), w, 1e-6)
    gy, gz = torch.randn_like(out), torch.randn_like(z)
    dx, dr, dw = torch.autograd.grad((out, z), (x, r, w), (gy, gz))

    xc = x.detach().clone().requires_grad_()
    rc = r.detach().clone().requires_grad_()
    wc = w.detach().clone().requires_grad_()
    zc = xc + rc
    outc = F.rms_norm(zc, (256,), wc, 1e-6)
    dxr, drr, dwr = torch.autograd.grad((outc, zc), (xc, rc, wc), (gy, gz))
    tol = GRAD_TOL[torch.float32]
    torch.testing.assert_close(dx, dxr, **tol)
    torch.testing.assert_close(dr, drr, **tol)
    torch.testing.assert_close(dw, dwr, **tol)


# --------------------------------------------------------------------------- #
# Structure: determinism, partial requires_grad, compiled backward.
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
def test_param_grads_bitwise_deterministic() -> None:
    """dgamma/dbeta use fixed-chunk partials + a fixed-shape aten sum, never
    atomics: two identical backwards must agree bit for bit."""
    x, _, w, b = _grad_inputs((4096, 1024), torch.float32, residual=False)
    gy = torch.randn(4096, 1024, device=x.device)

    def grads():
        y = layer_norm(x, (1024,), w, b, 1e-5)
        return torch.autograd.grad(y, (x, w, b), gy)

    dx1, dw1, db1 = grads()
    dx2, dw2, db2 = grads()
    assert torch.equal(dw1, dw2) and torch.equal(db1, db2) and torch.equal(dx1, dx2)


@pytest.mark.cuda
@requires_cuda_ext
def test_partial_requires_grad_combinations() -> None:
    x = _randn((8, 128), torch.float32)
    w, b = _affine(128, torch.float32)
    # weight-only grad
    wg = w.clone().requires_grad_()
    y = layer_norm(x, (128,), wg, b, 1e-5)
    (dw,) = torch.autograd.grad(y.sum(), (wg,))
    assert dw.shape == wg.shape
    # input-only grad, weightless call
    xg = x.clone().requires_grad_()
    y = rms_norm(xg, (128,), None, 1e-6)
    (dx,) = torch.autograd.grad(y.sum(), (xg,))
    assert dx.shape == xg.shape


@pytest.mark.cuda
@requires_cuda_ext
def test_compiled_backward() -> None:
    x = _randn((64, 1024), torch.float32).requires_grad_()
    w, b = _affine(1024, torch.float32)
    w.requires_grad_()

    def fn(x, w, b):
        return layer_norm(x, (1024,), w, b, 1e-5).sum()

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
    loss_e = fn(x, w, b)
    dx_e, dw_e = torch.autograd.grad(loss_e, (x, w))
    loss_c = compiled(x, w, b)
    dx_c, dw_c = torch.autograd.grad(loss_c, (x, w))
    torch.testing.assert_close(dx_c, dx_e)
    torch.testing.assert_close(dw_c, dw_e)


# --------------------------------------------------------------------------- #
# v0.5.0: vectorized-backward coverage, chunk-policy determinism, GELU bwd.
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
def test_backward_odd_n_large_m_and_misaligned() -> None:
    """The scalar backward path stays load-bearing in default mode: odd N at
    large M (vec ineligible), and a 16-byte-misaligned cotangent/input slice
    (contiguous but storage-offset) must fall back correctly, not crash."""
    x = _randn((4096, 1023), torch.float16).requires_grad_()
    w, _ = _affine(1023, torch.float16)
    w.requires_grad_()
    gy = torch.randn(4096, 1023, device=x.device, dtype=torch.float16)
    y = rms_norm(x, (1023,), w, 1e-6)
    dx, dw = torch.autograd.grad(y, (x, w), gy)
    xr = x.detach().float().requires_grad_()
    wr = w.detach().float().requires_grad_()
    yr = F.rms_norm(xr, (1023,), wr, 1e-6)
    dxr, dwr = torch.autograd.grad(yr, (xr, wr), gy.float())
    torch.testing.assert_close(dx.float(), dxr, **GRAD_TOL[torch.float16])
    torch.testing.assert_close(dw.float(), dwr, **GRAD_TOL[torch.float16])

    # misaligned: base[1:] keeps a storage offset ⇒ data_ptr not 16B-aligned
    base = torch.randn(512 * 1024 + 1, device=x.device)
    xm = base[1:].view(512, 1024).requires_grad_()
    assert xm.data_ptr() % 16 != 0
    y = rms_norm(xm, (1024,), None, 1e-6)
    (dxm,) = torch.autograd.grad(y.sum(), (xm,))
    xr = xm.detach().clone().requires_grad_()
    (dxr,) = torch.autograd.grad(F.rms_norm(xr, (1024,), None, 1e-6).sum(), (xr,))
    torch.testing.assert_close(dxm, dxr, **GRAD_TOL[torch.float32])


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("shape", [(512, 1024), (4096, 1024)],
                         ids=lambda s: "x".join(map(str, s)))
def test_param_grads_bitwise_deterministic_rms(shape) -> None:
    """RMS variant of the determinism contract, including the small-M shape
    where the GPU-filling chunk policy chooses a different chunk count."""
    x = _randn(shape, torch.float32).requires_grad_()
    w, _ = _affine(shape[-1], torch.float32)
    w.requires_grad_()
    gy = torch.randn(*shape, device=x.device)

    def grads():
        return torch.autograd.grad(rms_norm(x, (shape[-1],), w, 1e-6), (x, w), gy)

    dx1, dw1 = grads()
    dx2, dw2 = grads()
    assert torch.equal(dw1, dw2) and torch.equal(dx1, dx2)


@pytest.mark.cuda
@requires_cuda_ext
def test_partial_param_grads_through_finalize() -> None:
    """dgamma-only and dbeta-only requests exercise the single-launch
    finalize kernel's null-output legs."""
    x = _randn((64, 256), torch.float32)
    w, b = _affine(256, torch.float32)
    # dbeta only
    bg = b.clone().requires_grad_()
    (db,) = torch.autograd.grad(layer_norm(x, (256,), w, bg, 1e-5).sum(), (bg,))
    xr = x.clone()
    br = b.clone().requires_grad_()
    (dbr,) = torch.autograd.grad(F.layer_norm(xr, (256,), w, br, 1e-5).sum(), (br,))
    torch.testing.assert_close(db, dbr, **GRAD_TOL[torch.float32])
    # dgamma only
    wg = w.clone().requires_grad_()
    (dw,) = torch.autograd.grad(layer_norm(x, (256,), wg, b, 1e-5).sum(), (wg,))
    wr = w.clone().requires_grad_()
    (dwr,) = torch.autograd.grad(F.layer_norm(x, (256,), wr, b, 1e-5).sum(), (wr,))
    torch.testing.assert_close(dw, dwr, **GRAD_TOL[torch.float32])


# --------------------------------------------------------------------------- #
# GELU backward (v0.5.0): layer_norm_gelu no longer falls back under grad.
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("shape", [(3, 17), (2, 5, 32)], ids=lambda s: "x".join(map(str, s)))
def test_gradcheck_layer_norm_gelu(shape, approximate: str) -> None:
    from fused_layernorm import layer_norm_gelu

    n = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=torch.float64, requires_grad=True)
    w = (torch.rand(n, device="cuda", dtype=torch.float64) + 0.5).requires_grad_()
    b = torch.randn(n, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, w, b: layer_norm_gelu(x, (n,), w, b, 1e-5, approximate=approximate),
        (x, w, b), eps=1e-6, atol=1e-5,
    )


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_layer_norm_gelu_grads_match_composite(dtype, approximate: str) -> None:
    from fused_layernorm import layer_norm_gelu

    x, _, w, b = _grad_inputs((64, 1024), dtype, residual=False)
    gy = torch.randn(64, 1024, device=x.device, dtype=dtype)
    y = layer_norm_gelu(x, (1024,), w, b, 1e-5, approximate=approximate)
    assert y.grad_fn is not None  # fused train path, not the eager fallback
    dx, dw, db = torch.autograd.grad(y, (x, w, b), gy)
    xr = x.detach().float().requires_grad_()
    wr = w.detach().float().requires_grad_()
    br = b.detach().float().requires_grad_()
    yr = F.gelu(F.layer_norm(xr, (1024,), wr, br, 1e-5), approximate=approximate)
    dxr, dwr, dbr = torch.autograd.grad(yr, (xr, wr, br), gy.float())
    torch.testing.assert_close(dx.float(), dxr, **GRAD_TOL[dtype])
    torch.testing.assert_close(dw.float(), dwr, **GRAD_TOL[dtype])
    torch.testing.assert_close(db.float(), dbr, **GRAD_TOL[dtype])


@pytest.mark.cuda
@requires_cuda_ext
def test_layer_norm_gelu_param_grads_deterministic() -> None:
    from fused_layernorm import layer_norm_gelu

    x, _, w, b = _grad_inputs((512, 1024), torch.float32, residual=False)
    gy = torch.randn(512, 1024, device=x.device)

    def grads():
        y = layer_norm_gelu(x, (1024,), w, b, 1e-5, approximate="tanh")
        return torch.autograd.grad(y, (x, w, b), gy)

    dx1, dw1, db1 = grads()
    dx2, dw2, db2 = grads()
    assert torch.equal(dw1, dw2) and torch.equal(db1, db2) and torch.equal(dx1, dx2)
