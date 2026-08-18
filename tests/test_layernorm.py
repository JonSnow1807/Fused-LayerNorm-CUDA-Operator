"""Tests for the ``fused_layernorm`` package and the ``fused_layernorm_cuda`` extension.

Two groups of tests live here:

* **CPU tests** (always run): the pure-PyTorch fallback of ``layer_norm`` /
  ``layer_norm_gelu``, the ``LayerNorm`` module, ``LayerNorm.from_torch``
  parameter sharing, ``replace_layernorm``, and importability of the package
  without the compiled extension.
* **GPU tests** (marked ``cuda`` and skipped unless the extension is importable
  and a CUDA device exists): the kernel itself, always compared against
  ``torch.nn.functional.layer_norm`` with *random, non-identity* weight and
  bias.  Every tolerance is a numerical-agreement bound; there are no
  performance assertions anywhere in this file.

``FUSED_LN_TEST_DEVICE`` (default ``"cuda"``) selects the device the GPU-group
tests allocate on.  Setting it to ``"cpu"`` together with a stand-in
``fused_layernorm_cuda`` module on ``PYTHONPATH`` lets the *test logic* be
exercised on a machine without a GPU; it does not test the kernel.  A few tests
that need a real CUDA device (side streams, "CPU input is rejected") are skipped
in that mode.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from typing import Iterator, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_layernorm
from fused_layernorm import LayerNorm, layer_norm, layer_norm_gelu, replace_layernorm

# --------------------------------------------------------------------------- #
# Skip logic
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Numerical-agreement tolerances vs. F.layer_norm.  fp16 / bf16 outputs are
# compared against F.layer_norm computed in fp32 and cast back (PyTorch also
# accumulates in fp32 for those dtypes); the rtol values for those two are
# torch.testing.assert_close's own dtype defaults.
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


@contextlib.contextmanager
def _no_mha_fastpath() -> Iterator[None]:
    """Turn off PyTorch's fused TransformerEncoderLayer fast path while active.

    In eval mode that fast path reads ``norm1``/``norm2``'s parameters directly
    and never calls their ``forward``; disabling it makes sure a replaced
    ``LayerNorm`` module is actually exercised.
    """
    mha = getattr(torch.backends, "mha", None)
    if mha is None:  # torch < 2.2 has no such switch
        yield
        return
    prev = mha.get_fastpath_enabled()
    mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        mha.set_fastpath_enabled(prev)


# --------------------------------------------------------------------------- #
# CPU tests: package import, fallback path, module, replace_layernorm
# --------------------------------------------------------------------------- #


def test_package_exports_and_version() -> None:
    assert fused_layernorm.__version__ == "0.2.0"
    for name in ("layer_norm", "layer_norm_gelu", "LayerNorm", "replace_layernorm", "is_available"):
        assert hasattr(fused_layernorm, name)
    assert isinstance(fused_layernorm.is_available(), bool)
    if not torch.cuda.is_available():
        assert fused_layernorm.is_available() is False


def test_import_works_without_extension() -> None:
    """Importing the package must not require the compiled extension."""
    code = (
        "import sys; sys.modules['fused_layernorm_cuda'] = None\n"  # forces ImportError
        "import fused_layernorm\n"
        "assert fused_layernorm.layernorm._ext is None\n"
        "assert fused_layernorm.is_available() is False\n"
        "import torch\n"
        "x = torch.randn(3, 8)\n"
        "y = fused_layernorm.layer_norm(x, (8,))\n"
        "assert torch.equal(y, torch.nn.functional.layer_norm(x, (8,)))\n"
        "print('ok')\n"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_fallback_matches_torch(dtype: torch.dtype) -> None:
    x = torch.randn(4, 5, 16, dtype=dtype)
    w, b = _affine(16, dtype, device="cpu")
    for weight, bias in ((w, b), (w, None), (None, b), (None, None)):
        y = layer_norm(x, (16,), weight, bias, eps=1e-6)
        assert torch.equal(y, F.layer_norm(x, (16,), weight, bias, 1e-6))
    # int normalized_shape (as nn.LayerNorm accepts) and torch.Size are normalised
    # to a tuple; F.layer_norm itself only takes a sequence
    assert torch.equal(layer_norm(x, 16), F.layer_norm(x, (16,)))
    assert torch.equal(layer_norm(x, x.shape[-1:]), F.layer_norm(x, (16,)))
    # multi-dim normalized_shape is only ever handled by PyTorch
    w2 = torch.randn(5, 16, dtype=dtype)
    y2 = layer_norm(x, (5, 16), w2, None)
    assert torch.equal(y2, F.layer_norm(x, (5, 16), w2, None))


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_cpu_fallback_gelu_matches_torch(approximate: str) -> None:
    x = torch.randn(6, 32)
    w, b = _affine(32, torch.float32, device="cpu")
    y = layer_norm_gelu(x, (32,), w, b, approximate=approximate)
    expected = F.gelu(F.layer_norm(x, (32,), w, b), approximate=approximate)
    assert torch.equal(y, expected)


def test_cpu_fallback_never_calls_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU tensors must go to PyTorch even if the extension is importable."""

    class _Boom:
        def layernorm(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("extension called for a CPU tensor")

        layernorm_gelu = layernorm

    monkeypatch.setattr(fused_layernorm.layernorm, "_ext", _Boom())
    x = torch.randn(2, 8)
    layer_norm(x, (8,))
    layer_norm_gelu(x, (8,))
    with torch.inference_mode():
        layer_norm(x, (8,))


def test_cpu_fallback_gradients_flow() -> None:
    x = torch.randn(3, 8, requires_grad=True)
    w, b = _affine(8, torch.float32, device="cpu")
    w.requires_grad_()
    b.requires_grad_()
    y = layer_norm(x, (8,), w, b)
    assert y.requires_grad and y.grad_fn is not None
    y.square().sum().backward()
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    br = b.detach().clone().requires_grad_()
    F.layer_norm(xr, (8,), wr, br).square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(w.grad, wr.grad)
    torch.testing.assert_close(b.grad, br.grad)


def test_layernorm_module_matches_nn_layernorm() -> None:
    ref = nn.LayerNorm(24, eps=1e-6)
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        ref.bias.normal_()
    ours = LayerNorm(24, eps=1e-6)
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(2, 7, 24)
    assert torch.equal(ours(x), ref(x))
    assert list(ours.state_dict().keys()) == list(ref.state_dict().keys())
    # extra_repr / repr are inherited unchanged (the class is also called LayerNorm)
    assert ours.extra_repr() == ref.extra_repr()
    assert repr(ours) == repr(ref)
    assert isinstance(ours, nn.LayerNorm)
    # non-affine construction works too
    plain = LayerNorm(24, elementwise_affine=False)
    assert plain.weight is None and plain.bias is None
    assert torch.equal(plain(x), F.layer_norm(x, (24,)))


def test_from_torch_shares_parameters() -> None:
    orig = nn.LayerNorm(16, eps=1e-3)
    orig.eval()
    with torch.no_grad():
        orig.weight.uniform_(0.5, 1.5)
        orig.bias.normal_()
    new = LayerNorm.from_torch(orig)
    assert type(new) is LayerNorm
    assert new.normalized_shape == orig.normalized_shape
    assert new.eps == orig.eps
    assert new.elementwise_affine is True
    assert new.training is False
    # same Parameter objects, not clones
    assert new.weight is orig.weight
    assert new.bias is orig.bias
    assert dict(new.named_parameters()).keys() == dict(orig.named_parameters()).keys()
    # in-place updates on one side are visible on the other
    with torch.no_grad():
        orig.weight.add_(1.0)
    x = torch.randn(3, 16)
    assert torch.equal(new(x), orig(x))
    # optimizer holding the original parameters still trains the new module
    opt = torch.optim.SGD(orig.parameters(), lr=0.1)
    new(x).sum().backward()
    before = orig.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, orig.weight.detach())
    assert new.weight is orig.weight


def test_from_torch_non_affine_and_bias_free() -> None:
    plain = nn.LayerNorm(8, elementwise_affine=False)
    new = LayerNorm.from_torch(plain)
    assert new.elementwise_affine is False
    assert new.weight is None and new.bias is None
    x = torch.randn(2, 8)
    assert torch.equal(new(x), plain(x))
    assert new.state_dict().keys() == plain.state_dict().keys()

    # nn.LayerNorm(bias=False) exists since torch 2.1
    try:
        no_bias = nn.LayerNorm(8, bias=False)
    except TypeError:
        pytest.skip("this torch version has no nn.LayerNorm(bias=...) argument")
    new = LayerNorm.from_torch(no_bias)
    assert new.weight is no_bias.weight
    assert new.bias is None
    assert torch.equal(new(x), no_bias(x))

    with pytest.raises(TypeError):
        LayerNorm.from_torch(nn.Linear(8, 8))  # type: ignore[arg-type]


class _CustomLN(nn.LayerNorm):
    """A user subclass; replace_layernorm must leave it alone."""


def _small_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(16, 16),
        nn.LayerNorm(16),
        nn.ModuleList([nn.LayerNorm(16), nn.Sequential(nn.ReLU(), nn.LayerNorm(16))]),
        nn.ModuleDict({"a": nn.LayerNorm(16), "b": nn.Linear(16, 16)}),
        _CustomLN(16),
        nn.LayerNorm((4, 16)),  # 2-D normalized_shape: not replaced
    )
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_()
    return model


def test_replace_layernorm_on_small_model() -> None:
    model = _small_model()
    x = torch.randn(4, 16)

    def run(m: nn.Module) -> torch.Tensor:
        h = m[1](m[0](x))
        h = m[2][1](m[2][0](h))
        h = m[3]["b"](m[3]["a"](h))
        h = m[4](h)
        return m[5](h.view(1, 4, 16))

    expected = run(model)
    param_ids = {id(p) for p in model.parameters()}
    keys = list(model.state_dict().keys())

    count = replace_layernorm(model)
    assert count == 4  # Sequential, ModuleList, nested Sequential, ModuleDict
    assert type(model[1]) is LayerNorm
    assert type(model[2][0]) is LayerNorm
    assert type(model[2][1][1]) is LayerNorm
    assert type(model[3]["a"]) is LayerNorm
    assert type(model[4]) is _CustomLN  # subclass untouched
    assert type(model[5]) is nn.LayerNorm  # multi-dim untouched
    assert type(model[0]) is nn.Linear

    # parameters are shared and the state_dict layout is unchanged
    assert {id(p) for p in model.parameters()} == param_ids
    assert list(model.state_dict().keys()) == keys
    assert torch.equal(run(model), expected)
    # idempotent: our LayerNorm is a subclass, not exact nn.LayerNorm
    assert replace_layernorm(model) == 0


def test_replace_layernorm_root_and_containers() -> None:
    # the root module itself is never replaced
    root = nn.LayerNorm(8)
    assert replace_layernorm(root) == 0 and type(root) is nn.LayerNorm
    # containers as roots
    lst = nn.ModuleList([nn.LayerNorm(8), nn.Linear(8, 8), nn.LayerNorm(8)])
    assert replace_layernorm(lst) == 2
    assert all(type(m) is LayerNorm for m in (lst[0], lst[2]))
    dct = nn.ModuleDict({"x": nn.LayerNorm(8)})
    assert replace_layernorm(dct) == 1 and type(dct["x"]) is LayerNorm
    assert replace_layernorm(nn.Linear(2, 2)) == 0
    # a standard transformer layer has exactly two LayerNorms
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
    assert replace_layernorm(layer) == 2
    assert type(layer.norm1) is LayerNorm and type(layer.norm2) is LayerNorm


def test_replace_layernorm_does_not_touch_torch_nn() -> None:
    model = _small_model()
    replace_layernorm(model)
    assert nn.LayerNorm is torch.nn.modules.normalization.LayerNorm
    assert type(nn.LayerNorm(4)) is nn.LayerNorm


# --------------------------------------------------------------------------- #
# GPU tests: the kernel, vs F.layer_norm with random non-identity affine
# --------------------------------------------------------------------------- #


@pytest.mark.cuda
@requires_cuda_ext
def test_extension_version() -> None:
    assert _ext.__version__ == fused_layernorm.__version__


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_kernel_matches_layer_norm(shape: Tuple[int, int], dtype: torch.dtype) -> None:  # (1)
    x = _randn(shape, dtype)
    w, b = _affine(shape[-1], dtype)
    y = _ext.layernorm(x, w, b, 1e-5)
    _assert_close(y, _ref_layer_norm(x, w, b, 1e-5))
    assert y.data_ptr() != x.data_ptr()


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_float64_ill_conditioned_rows() -> None:  # (1)
    """Rows with mean 1000 and std 1e-3: statistics must be accumulated in double.

    With float32 accumulation the mean alone is off by roughly the ulp of 1000
    in fp32 (about 1e-4), which after dividing by std 1e-3 becomes an output
    error of order 1e-2..1e-1.  Two correct float64 implementations that reduce
    in a different order still disagree here by an amount of order 1e-10
    (analytical estimate: mean/std = 1e6 amplifies double rounding of the mean;
    not a measured figure), so the bound below is deliberately looser than the
    well-conditioned float64 tolerance in ``TOL`` (a spec deviation), while
    remaining many orders of magnitude below the fp32-accumulation failure mode
    this test guards against.
    """
    x = torch.randn(8, 4096, dtype=torch.float64, device=DEVICE) * 1e-3 + 1000.0
    w, b = _affine(4096, torch.float64)
    y = _ext.layernorm(x, w, b, 1e-5)
    expected = F.layer_norm(x, (4096,), w, b, 1e-5)
    assert y.dtype == torch.float64
    torch.testing.assert_close(y, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
@pytest.mark.parametrize("which", ["weight_only", "bias_only", "neither"])
def test_kernel_optional_affine(which: str, dtype: torch.dtype) -> None:  # (2)
    x = _randn((16, 1024), dtype)
    w, b = _affine(1024, dtype)
    weight = w if which == "weight_only" else None
    bias = b if which == "bias_only" else None
    y = _ext.layernorm(x, weight, bias, 1e-5)
    _assert_close(y, _ref_layer_norm(x, weight, bias, 1e-5))
    # keyword form and defaults (weight=None, bias=None, eps=1e-5)
    y_kw = _ext.layernorm(x, weight=weight, bias=bias)
    _assert_close(y_kw, _ref_layer_norm(x, weight, bias, 1e-5))
    _assert_close(_ext.layernorm(x), _ref_layer_norm(x, None, None, 1e-5))


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("shape", [(768,), (5, 3, 256), (2, 3, 4, 768)], ids=str)
def test_kernel_higher_rank_and_rank1(shape: Tuple[int, ...]) -> None:  # (3)
    x = _randn(shape, torch.float32)
    w, b = _affine(shape[-1], torch.float32)
    y = _ext.layernorm(x, w, b, 1e-5)
    assert y.shape == x.shape
    _assert_close(y, F.layer_norm(x, (shape[-1],), w, b, 1e-5))


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_custom_eps() -> None:
    x = _randn((8, 512), torch.float32) * 1e-3
    w, b = _affine(512, torch.float32)
    for eps in (1e-1, 1e-3, 1e-8):
        _assert_close(_ext.layernorm(x, w, b, eps), F.layer_norm(x, (512,), w, b, eps))


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_non_contiguous_input() -> None:  # (4)
    m, n = 64, 512
    w, b = _affine(n, torch.float32)
    # transposed allocation
    x_t = _randn((n, m), torch.float32).t()
    assert not x_t.is_contiguous() and x_t.shape == (m, n)
    _assert_close(_ext.layernorm(x_t, w, b, 1e-5), F.layer_norm(x_t, (n,), w, b, 1e-5))
    # strided column slice
    x_s = _randn((m, 2 * n), torch.float32)[:, ::2]
    assert not x_s.is_contiguous() and x_s.shape == (m, n)
    _assert_close(_ext.layernorm(x_s, w, b, 1e-5), F.layer_norm(x_s, (n,), w, b, 1e-5))
    # non-contiguous weight / bias
    w_s = _randn((2 * n,), torch.float32)[::2]
    b_s = _randn((2 * n,), torch.float32)[1::2]
    x = _randn((m, n), torch.float32)
    _assert_close(_ext.layernorm(x, w_s, b_s, 1e-5), F.layer_norm(x, (n,), w_s, b_s, 1e-5))


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_argument_errors() -> None:  # (5)
    x = _randn((4, 32), torch.float32)
    w, b = _affine(32, torch.float32)
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, _affine(31, torch.float32)[0], b)  # wrong weight length
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, w, _affine(33, torch.float32)[1])  # wrong bias length
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, w.view(4, 8), b)  # weight not 1-D
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, w.double(), b)  # weight dtype mismatch
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, w, b.half())  # bias dtype mismatch
    with pytest.raises(RuntimeError):
        _ext.layernorm(torch.tensor(1.0, device=DEVICE))  # rank-0 input
    with pytest.raises(RuntimeError):
        _ext.layernorm_gelu(x, w, b, 1e-5, "foo")  # bad approximate
    with pytest.raises(RuntimeError):
        _ext.layernorm_gelu(x, w, b, approximate="erf")


@pytest.mark.cuda
@requires_cuda_ext
@requires_real_cuda
def test_kernel_rejects_cpu_input() -> None:  # (5)
    x = torch.randn(4, 32)
    w, b = _affine(32, torch.float32, device="cpu")
    with pytest.raises(RuntimeError):
        _ext.layernorm(x, w, b)
    # device mismatch between input and affine
    with pytest.raises(RuntimeError):
        _ext.layernorm(x.cuda(), w, b)


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_kernel_gelu(approximate: str, dtype: torch.dtype) -> None:  # (6)
    x = _randn((32, 1024), dtype)
    w, b = _affine(1024, dtype)
    y = _ext.layernorm_gelu(x, w, b, 1e-5, approximate)
    expected = _ref_layer_norm(x, w, b, 1e-5, approximate=approximate)
    _assert_close(y, expected)
    # keyword form and default approximate="none"
    _assert_close(_ext.layernorm_gelu(x, w, b, approximate=approximate), expected)
    if approximate == "none":
        _assert_close(_ext.layernorm_gelu(x, w, b), expected)


@pytest.mark.cuda
@requires_cuda_ext
@requires_real_cuda
def test_kernel_side_stream() -> None:  # (7)
    """Inputs produced and consumed on a non-default stream must be handled correctly."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(256, 4096, device="cuda")
        w = torch.empty(4096, device="cuda").uniform_(0.5, 1.5)
        b = torch.randn(4096, device="cuda")
        y = _ext.layernorm(x, w, b, 1e-5)
    torch.cuda.synchronize()
    _assert_close(y, F.layer_norm(x, (4096,), w, b, 1e-5))


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_deterministic() -> None:  # (8)
    x = _randn((64, 4096), torch.float32)
    w, b = _affine(4096, torch.float32)
    y1 = _ext.layernorm(x, w, b, 1e-5)
    y2 = _ext.layernorm(x, w, b, 1e-5)
    assert torch.equal(y1, y2)
    g1 = _ext.layernorm_gelu(x, w, b, 1e-5, "tanh")
    g2 = _ext.layernorm_gelu(x, w, b, 1e-5, "tanh")
    assert torch.equal(g1, g2)


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_has_no_grad_fn() -> None:  # (9)
    """The extension is forward-only: its output never carries a grad_fn."""
    x = _randn((4, 64), torch.float32).requires_grad_()
    w, b = _affine(64, torch.float32)
    w.requires_grad_()
    y = _ext.layernorm(x, w, b, 1e-5)
    assert y.requires_grad is False
    assert y.grad_fn is None
    yg = _ext.layernorm_gelu(x, w, b, 1e-5, "none")
    assert yg.requires_grad is False and yg.grad_fn is None


@pytest.mark.cuda
@requires_cuda_ext
def test_wrapper_falls_back_to_torch_when_grad_needed() -> None:  # (9)
    x = _randn((4, 64), torch.float32).requires_grad_()
    w, b = _affine(64, torch.float32)
    w.requires_grad_()
    b.requires_grad_()
    y = layer_norm(x, (64,), w, b)
    assert y.requires_grad and y.grad_fn is not None
    y.sum().backward()
    assert x.grad is not None and w.grad is not None and b.grad is not None
    yg = layer_norm_gelu(x, (64,), w, b, approximate="tanh")
    assert yg.requires_grad and yg.grad_fn is not None
    # ... and only inputs requiring grad matter: detached inputs under grad
    # mode are still eligible for the kernel, hence no grad_fn.
    y_det = layer_norm(x.detach(), (64,), w.detach(), b.detach())
    assert y_det.grad_fn is None
    with torch.no_grad():
        assert layer_norm(x, (64,), w, b).grad_fn is None
    with torch.inference_mode():
        assert layer_norm(x, (64,), w, b).grad_fn is None


@pytest.mark.cuda
@requires_cuda_ext
@requires_real_cuda
def test_wrapper_dispatches_to_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper takes the fused path exactly when the documented rule holds."""
    calls = []

    class _Spy:
        def layernorm(self, *args, **kwargs):
            calls.append("layernorm")
            return _ext.layernorm(*args, **kwargs)

        def layernorm_gelu(self, *args, **kwargs):
            calls.append("layernorm_gelu")
            return _ext.layernorm_gelu(*args, **kwargs)

    monkeypatch.setattr(fused_layernorm.layernorm, "_ext", _Spy())
    x = _randn((4, 5, 64), torch.float32)
    w, b = _affine(64, torch.float32)
    with torch.inference_mode():
        y = layer_norm(x, (64,), w, b)
        yg = layer_norm_gelu(x, (64,), w, b, approximate="tanh")
        _assert_close(y, F.layer_norm(x, (64,), w, b))
        _assert_close(yg, F.gelu(F.layer_norm(x, (64,), w, b), approximate="tanh"))
        assert calls == ["layernorm", "layernorm_gelu"]
        # multi-dim normalized_shape and CPU tensors are not eligible
        layer_norm(x, (5, 64), None, None)
        layer_norm(x.cpu(), (64,), w.cpu(), b.cpu())
        assert calls == ["layernorm", "layernorm_gelu"]
        # module path
        mod = LayerNorm(64).to(DEVICE)
        mod(x)
        assert calls[-1] == "layernorm"
    # grad needed -> torch
    xr = x.clone().requires_grad_()
    layer_norm(xr, (64,), w, b)
    assert len(calls) == 3
    with torch.inference_mode():
        # mixed dtype (fp16 activations, fp32 parameters) -> torch, never the
        # kernel.  PyTorch's own CUDA LayerNorm rejects this outside autocast, so
        # the wrapper is expected to raise exactly as nn.LayerNorm would.
        with contextlib.suppress(RuntimeError):
            layer_norm(x.half(), (64,), w, b)
        with contextlib.suppress(RuntimeError):
            layer_norm_gelu(x.half(), (64,), w, b)
        # weight on another device than input -> torch (whatever F.layer_norm
        # then does with it is PyTorch's decision, not the kernel's)
        with contextlib.suppress(RuntimeError):
            layer_norm(x, (64,), w.cpu(), b)
        # under CUDA autocast -> torch, so the fp32-output semantics are kept
        with torch.autocast("cuda", dtype=torch.float16):
            layer_norm(x, (64,), w, b)
            layer_norm(x.half(), (64,), w, b)
            layer_norm_gelu(x, (64,), w, b)
            mod(x)
        assert len(calls) == 3
        # ... and the fused path is taken again once autocast is left
        layer_norm(x, (64,), w, b)
        assert len(calls) == 4


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16], ids=str)
def test_wrapper_mixed_dtype_and_autocast_match_torch(act_dtype: torch.dtype) -> None:
    """Mixed-dtype calls (low-precision activations, fp32 parameters) behave like PyTorch.

    The kernel rejects a dtype mismatch, so the wrapper hands these calls to
    PyTorch and must then do whatever ``F.layer_norm`` / ``nn.LayerNorm`` does:
    on CUDA, outside autocast, PyTorch's LayerNorm itself raises for a
    weight/bias dtype different from the input; under autocast it computes and
    returns fp32.  Either way ``replace_layernorm`` stays a drop-in.
    """
    n = 256
    x = _randn((4, 8, n), act_dtype)
    w, b = _affine(n, torch.float32)
    ref = nn.LayerNorm(n).to(DEVICE)
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        ref.bias.normal_()
    ours = LayerNorm.from_torch(ref)
    with torch.inference_mode():
        # Whatever PyTorch does here (raise on CUDA, compute on CPU), the
        # wrapper must do the same.
        try:
            expected = F.layer_norm(x, (n,), w, b)
        except RuntimeError:
            expected = None
        if expected is None:
            with pytest.raises(RuntimeError):
                layer_norm(x, (n,), w, b)
            with pytest.raises(RuntimeError):
                layer_norm_gelu(x, (n,), w, b, approximate="tanh")
            with pytest.raises(RuntimeError):
                ours(x)
        else:
            y = layer_norm(x, (n,), w, b)
            assert y.dtype == expected.dtype
            torch.testing.assert_close(y, expected)
            yg = layer_norm_gelu(x, (n,), w, b, approximate="tanh")
            torch.testing.assert_close(yg, F.gelu(expected, approximate="tanh"))
            torch.testing.assert_close(ours(x), ref(x))
        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=act_dtype):
                y_ac = layer_norm(x.float(), (n,), w, b)
                exp_ac = F.layer_norm(x.float(), (n,), w, b)
                assert y_ac.dtype == exp_ac.dtype
                torch.testing.assert_close(y_ac, exp_ac)
                torch.testing.assert_close(ours(x), ref(x))


@pytest.mark.cuda
@requires_cuda_ext
def test_layernorm_module_on_device_matches_torch() -> None:
    ref = nn.LayerNorm(768).to(DEVICE)
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        ref.bias.normal_()
    ours = LayerNorm.from_torch(ref)
    x = _randn((3, 7, 768), torch.float32)
    with torch.inference_mode():
        _assert_close(ours(x), ref(x))
    layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
    layer = layer.to(DEVICE).eval()
    src = _randn((2, 9, 64), torch.float32)
    with _no_mha_fastpath(), torch.inference_mode():
        expected = layer(src)
        assert replace_layernorm(layer) == 2
        _assert_close(layer(src), expected)


@pytest.mark.cuda
@requires_cuda_ext
def test_kernel_empty_input() -> None:  # (10)
    w, b = _affine(768, torch.float32)
    x = torch.empty(0, 768, device=DEVICE)
    y = _ext.layernorm(x, w, b, 1e-5)
    assert y.shape == (0, 768) and y.dtype == x.dtype and y.device == x.device
    yg = _ext.layernorm_gelu(x, w, b, 1e-5, "tanh")
    assert yg.shape == (0, 768)
    x3 = torch.empty(2, 0, 768, device=DEVICE, dtype=torch.float16)
    assert _ext.layernorm(x3, w.half(), b.half()).shape == (2, 0, 768)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
