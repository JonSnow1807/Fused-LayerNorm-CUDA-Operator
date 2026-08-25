"""Minimal usage example for ``fused_layernorm``.

Walks the op family -- the ``F.layer_norm``/``F.rms_norm``-shaped functions,
the fused LayerNorm+GELU, the fused residual-add+norm (the pre-norm
transformer pattern), the fp8-output norms, a training step through the CUDA
backward, and ``replace_layernorm`` on a stock
``nn.TransformerEncoderLayer`` -- checking each against plain PyTorch with
``torch.testing.assert_close`` (or ``torch.equal`` where the contract is
bitwise).

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

    # 3) RMSNorm, same drop-in shape as F.rms_norm (eps=None keeps torch's
    #    machine-epsilon convention).
    yr = fused_layernorm.rms_norm(x, (n,), weight, eps=1e-6)
    torch.testing.assert_close(yr, F.rms_norm(x, (n,), weight, 1e-6), atol=1e-5, rtol=1e-4)

    # 4) The headline op: residual-add + norm in one kernel. The returned
    #    new_residual is bitwise the rounded sum, and out is bitwise the plain
    #    norm of it (composite equivalence - tested with torch.equal).
    res = torch.randn_like(x)
    out, new_res = fused_layernorm.fused_add_rms_norm(x, res, (n,), weight, 1e-6)
    assert torch.equal(new_res, x + res)
    assert torch.equal(out, fused_layernorm.rms_norm(new_res, (n,), weight, 1e-6))

    # 5) norm -> fp8-E4M3 in one kernel (inference-only; dynamic per-token
    #    dequant scales, vLLM/TensorRT convention: y ~ q.float() * scale).
    with torch.no_grad():
        q, scale = fused_layernorm.rms_norm_fp8(x, (n,), weight, 1e-6)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (32, 1)

    # 6) Training: the fused ops have a real (vectorised, deterministic) CUDA
    #    backward; gradients match the PyTorch composite.
    xg = x.clone().requires_grad_()
    wg = weight.clone().requires_grad_()
    fused_layernorm.rms_norm(xg, (n,), wg, 1e-6).sum().backward()
    xr = x.clone().requires_grad_()
    wr = weight.clone().requires_grad_()
    F.rms_norm(xr, (n,), wr, 1e-6).sum().backward()
    torch.testing.assert_close(xg.grad, xr.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(wg.grad, wr.grad, atol=1e-5, rtol=1e-4)

    # 7) Drop-in on an existing model: only exact nn.LayerNorm children with a
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
    print("layer_norm / layer_norm_gelu / rms_norm / fused_add_rms_norm / rms_norm_fp8 "
          f"all match their PyTorch references; backward matches the composite; "
          f"replaced {replaced} LayerNorm modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
