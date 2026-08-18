"""Minimal usage example for ``fused_layernorm``.

Shows the three entry points -- the ``F.layer_norm``-shaped function, the fused
LayerNorm+GELU function, and ``replace_layernorm`` on a stock
``nn.TransformerEncoderLayer`` -- and checks each against plain PyTorch with
``torch.testing.assert_close``.  Everything is forward-only (the kernel has no
backward pass), so the module demo runs under ``torch.inference_mode()``.

Requires the compiled ``fused_layernorm_cuda`` extension and a CUDA device;
otherwise it prints a message and exits 0.  Run with ``python examples/example_usage.py``.
"""

import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_layernorm


def main() -> int:
    if not fused_layernorm.is_available():
        print(
            "fused_layernorm_cuda is not built or CUDA is unavailable; nothing to run.\n"
            "On a machine with CUDA and nvcc, build with `pip install --no-build-isolation -e .` from the repo root."
        )
        return 0

    dev = torch.device("cuda")
    n = 768
    x = torch.randn(32, n, device=dev)
    weight = torch.empty(n, device=dev).uniform_(0.5, 1.5)
    bias = torch.randn(n, device=dev)

    # 1) Functional call: identical signature to F.layer_norm.
    y = fused_layernorm.layer_norm(x, (n,), weight, bias, eps=1e-5)
    torch.testing.assert_close(y, F.layer_norm(x, (n,), weight, bias, 1e-5), atol=1e-5, rtol=1e-4)

    # 2) Fused LayerNorm + GELU (erf or tanh form).
    yg = fused_layernorm.layer_norm_gelu(x, (n,), weight, bias, approximate="tanh")
    ref = F.gelu(F.layer_norm(x, (n,), weight, bias), approximate="tanh")
    torch.testing.assert_close(yg, ref, atol=1e-5, rtol=1e-4)

    # 3) Drop-in on an existing model: only exact nn.LayerNorm children with a
    #    1-D normalized_shape are swapped, parameters are shared, nothing global
    #    is patched.
    layer = nn.TransformerEncoderLayer(d_model=n, nhead=12, batch_first=True).to(dev).eval()
    fused = copy.deepcopy(layer)
    replaced = fused_layernorm.replace_layernorm(fused)
    src = torch.randn(4, 16, n, device=dev)
    # Demo-only detail: in eval mode PyTorch's TransformerEncoderLayer takes a
    # fused "fast path" that reads norm1/norm2's parameters directly and never
    # calls their forward().  It is switched off here (and restored afterwards)
    # only so that the replaced LayerNorm modules are actually exercised in
    # this comparison; your own code does not need to do this.
    mha = getattr(torch.backends, "mha", None)
    prev_fastpath = mha.get_fastpath_enabled() if mha is not None else None
    try:
        if mha is not None:
            mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            out_ref = layer(src)
            out = fused(src)
    finally:
        if mha is not None:
            mha.set_fastpath_enabled(prev_fastpath)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

    print(f"fused_layernorm {fused_layernorm.__version__} on {torch.cuda.get_device_name(dev)}")
    print(f"layer_norm / layer_norm_gelu match F.layer_norm; replaced {replaced} LayerNorm modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
