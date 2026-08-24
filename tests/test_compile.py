"""torch.compile integration: the public wrappers must trace with NO graph
breaks (fullgraph) and match eager numerics.

The custom-op layer exists precisely because raw pybind calls hard-graph-break
under Dynamo; these tests pin that property. The CPU test compiles the
PyTorch-fallback path (what the CPU-only CI can verify); the CUDA tests
compile the fused path end to end.
"""

from __future__ import annotations

import pytest
import torch

import fused_layernorm

from _helpers import _affine, _randn, requires_cuda_ext


def _explain_graph_breaks(fn, *args):
    explanation = torch._dynamo.explain(fn)(*args)
    return explanation.graph_break_count


def test_fallback_path_compiles_fullgraph_on_cpu() -> None:
    x = torch.randn(4, 64)
    w = torch.rand(64) + 0.5
    b = torch.randn(64)

    def fn(x, w, b):
        return fused_layernorm.layer_norm(x, (64,), w, b, 1e-5)

    torch._dynamo.reset()
    assert _explain_graph_breaks(fn, x, w, b) == 0
    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(x, w, b), fn(x, w, b))


@pytest.mark.cuda
@requires_cuda_ext
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fused_path_compiles_fullgraph(backend: str) -> None:
    x = _randn((64, 1024), torch.float32)
    w, b = _affine(1024, torch.float32)

    def fn(x, w, b):
        return fused_layernorm.layer_norm(x, (1024,), w, b, 1e-5)

    torch._dynamo.reset()
    assert _explain_graph_breaks(fn, x, w, b) == 0
    compiled = torch.compile(fn, fullgraph=True, backend=backend)
    torch.testing.assert_close(compiled(x, w, b), fn(x, w, b))


@pytest.mark.cuda
@requires_cuda_ext
def test_fused_gelu_path_compiles_fullgraph() -> None:
    x = _randn((32, 512), torch.float32)
    w, b = _affine(512, torch.float32)

    def fn(x, w, b):
        return fused_layernorm.layer_norm_gelu(x, (512,), w, b, 1e-5, approximate="tanh")

    torch._dynamo.reset()
    assert _explain_graph_breaks(fn, x, w, b) == 0
    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(x, w, b), fn(x, w, b))


@pytest.mark.cuda
@requires_cuda_ext
def test_module_compiles_fullgraph() -> None:
    m = fused_layernorm.LayerNorm(768).cuda().eval()
    x = _randn((8, 16, 768), torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(m, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), m(x))
