"""Tests for the torch.library custom-op layer (fused_layernorm::*).

Two groups:
* CPU-runnable: the ops are registered with the expected schemas whether or
  not the extension is built (registration is unconditional by design).
* CUDA-marked: ``torch.library.opcheck`` runs its full battery (schema
  correctness, fake-impl consistency, aot-dispatch) against the real kernels.
"""

from __future__ import annotations

import pytest
import torch

import fused_layernorm  # noqa: F401  (import registers the ops)

from _helpers import DEVICE, _affine, _randn, requires_cuda_ext

# Note: the eps default renders as repr(1e-5) = 1.0000000000000001e-05 in the
# schema string (float round-trip formatting), not "1e-05".
EXPECTED_SCHEMAS = {
    "layer_norm": (
        "fused_layernorm::layer_norm(Tensor input, Tensor? weight=None, "
        "Tensor? bias=None, float eps=1.0000000000000001e-05) -> Tensor"
    ),
    "layer_norm_gelu": (
        "fused_layernorm::layer_norm_gelu(Tensor input, Tensor? weight=None, "
        "Tensor? bias=None, float eps=1.0000000000000001e-05, "
        'str approximate="none") -> Tensor'
    ),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_SCHEMAS))
def test_op_registered_with_expected_schema(name: str) -> None:
    packet = getattr(torch.ops.fused_layernorm, name, None)
    assert packet is not None, f"torch.ops.fused_layernorm.{name} is not registered"
    assert str(packet.default._schema) == EXPECTED_SCHEMAS[name]


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("with_affine", [True, False])
def test_opcheck_layer_norm(with_affine: bool) -> None:
    x = _randn((8, 256), torch.float32)
    w, b = _affine(256, torch.float32) if with_affine else (None, None)
    torch.library.opcheck(
        torch.ops.fused_layernorm.layer_norm.default, (x, w, b, 1e-5)
    )


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_opcheck_layer_norm_gelu(approximate: str) -> None:
    x = _randn((8, 256), torch.float32)
    w, b = _affine(256, torch.float32)
    torch.library.opcheck(
        torch.ops.fused_layernorm.layer_norm_gelu.default, (x, w, b, 1e-5, approximate)
    )


@pytest.mark.cuda
@requires_cuda_ext
def test_op_matches_raw_pybind() -> None:
    """The dispatcher route and the raw extension call are the same kernel."""
    from fused_layernorm._common import _ext

    x = _randn((64, 1024), torch.float32)
    w, b = _affine(1024, torch.float32)
    via_op = torch.ops.fused_layernorm.layer_norm(x, w, b, 1e-5)
    via_ext = _ext.layernorm(x, w, b, 1e-5)
    assert torch.equal(via_op, via_ext)


@pytest.mark.cuda
@requires_cuda_ext
def test_fake_impl_shapes_under_fake_tensor_mode() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(4, 7, 128, device=DEVICE)
        y = torch.ops.fused_layernorm.layer_norm(x, None, None, 1e-5)
        assert y.shape == x.shape and y.dtype == x.dtype and y.is_contiguous()
