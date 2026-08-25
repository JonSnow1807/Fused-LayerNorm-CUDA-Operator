"""Randomized contract fuzz — run explicitly, not collected by pytest.

    PYTHONPATH=. python tests/fuzz_contracts.py [--rounds N] [--seed S]

Hammers the documented contracts at random shapes/dtypes the fixed-shape
suite cannot enumerate: the bitwise fused-add composite equivalence, plain
norm numerics vs the fp32 composite, the fp8 byte/scale contract (with NaN
injection — this fuzz found the v0.4.2 dynamic-scale NaN bug), and backward
numerics + determinism vs autograd. Exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

import fused_layernorm as fl

TOL = {
    torch.float32: dict(rtol=2e-5, atol=2e-5),
    torch.float16: dict(rtol=2e-3, atol=2e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}
BWD_TOL = {
    torch.float32: dict(rtol=5e-3, atol=5e-3),
    torch.float16: dict(rtol=5e-2, atol=5e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

fails = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global fails
    if not cond:
        fails += 1
        print(f"FAIL {name}: {detail}")


def shapes(n: int):
    """Random (M, N) biased toward awkward N (odd, non-multiple-of-vec)."""
    for _ in range(n):
        m = int(torch.randint(1, 5000, (1,)))
        ncand = [int(torch.randint(8, 8193, (1,))), 1024, 1536, 4096, 8192,
                 int(torch.randint(8, 8193, (1,))) | 1]
        yield m, ncand[int(torch.randint(0, len(ncand), (1,)))]


def fuzz_fused_add(rounds: int) -> None:
    for i, (m, n) in enumerate(shapes(rounds)):
        dt = DTYPES[i % 3]
        x = torch.randn(m, n, device="cuda", dtype=dt)
        r = torch.randn(m, n, device="cuda", dtype=dt)
        w = torch.rand(n, device="cuda", dtype=dt) + 0.5
        b = torch.randn(n, device="cuda", dtype=dt)
        zc = (x.float() + r.float()).to(dt)
        out, z = fl.fused_add_rms_norm(x, r, (n,), w, 1e-6)
        check("fused_add_rms z", torch.equal(z, zc), f"{m}x{n} {dt}")
        check("fused_add_rms out", torch.equal(out, fl.rms_norm(z, (n,), w, 1e-6)),
              f"{m}x{n} {dt}")
        out2, z2 = fl.fused_add_layer_norm(x, r, (n,), w, b, 1e-5)
        check("fused_add_ln z", torch.equal(z2, zc), f"{m}x{n} {dt}")
        check("fused_add_ln out", torch.equal(out2, fl.layer_norm(z2, (n,), w, b, 1e-5)),
              f"{m}x{n} {dt}")


def fuzz_forward_numerics(rounds: int) -> None:
    for i, (m, n) in enumerate(shapes(rounds)):
        dt = DTYPES[i % 3]
        x = torch.randn(m, n, device="cuda", dtype=dt) * (10 ** int(torch.randint(-2, 3, (1,))))
        w = torch.rand(n, device="cuda", dtype=dt) + 0.5
        for name, got, ref in [
            ("rms numeric", fl.rms_norm(x, (n,), w, 1e-6),
             F.rms_norm(x.float(), (n,), w.float(), 1e-6).to(dt)),
            ("ln numeric", fl.layer_norm(x, (n,), w, None, 1e-5),
             F.layer_norm(x.float(), (n,), w.float(), None, 1e-5).to(dt)),
        ]:
            try:
                torch.testing.assert_close(got, ref, **TOL[dt])
            except AssertionError as e:
                check(name, False, f"{m}x{n} {dt}: {str(e).splitlines()[0]}")


def _quant_ref(y: torch.Tensor, s) -> torch.Tensor:
    return (y.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def fuzz_fp8(rounds: int) -> None:
    for i, (m, n) in enumerate(shapes(rounds)):
        dt = DTYPES[i % 3]
        x = torch.randn(m, n, device="cuda", dtype=dt) * (10 ** int(torch.randint(-2, 4, (1,))))
        if i % 5 == 0 and m > 1:
            x[0, 0] = float("nan")  # the injection that found the v0.4.2 bug
        w = torch.rand(n, device="cuda", dtype=dt) + 0.5
        with torch.no_grad():
            y = fl.rms_norm(x, (n,), w, 1e-6)
            s = torch.rand(1, device="cuda") + 0.01
            o, _ = fl.rms_norm_fp8(x, (n,), w, 1e-6, scale=s)
            check("fp8 static bytes",
                  torch.equal(o.view(torch.uint8), _quant_ref(y, s).view(torch.uint8)),
                  f"{m}x{n} {dt}")
            o, s_dyn = fl.rms_norm_fp8(x, (n,), w, 1e-6)
            amax = y.float().abs().amax(-1, keepdim=True)
            s_ref = torch.where(torch.isnan(amax), amax, amax.clamp(min=1e-12)) / 448.0
            ok = ((s_dyn - s_ref).abs() <= s_ref * 1e-6 + 1e-12) | (
                torch.isnan(s_dyn) & torch.isnan(s_ref))
            check("fp8 dyn scale", bool(ok.all()), f"{m}x{n} {dt}")
            live = ~torch.isnan(s_dyn).squeeze(-1)
            check("fp8 dyn bytes",
                  torch.equal(o[live].view(torch.uint8),
                              _quant_ref(y[live], s_dyn[live]).view(torch.uint8)),
                  f"{m}x{n} {dt}")


def fuzz_backward(rounds: int) -> None:
    for i, (m, n) in enumerate(shapes(rounds)):
        dt = DTYPES[i % 3]
        m = min(m, 512)
        x = torch.randn(m, n, device="cuda", dtype=dt, requires_grad=True)
        w = (torch.rand(n, device="cuda", dtype=dt) + 0.5).requires_grad_()
        g = torch.randn(m, n, device="cuda", dtype=dt)
        dx, dw = torch.autograd.grad(fl.rms_norm(x, (n,), w, 1e-6), (x, w), g)
        xr = x.detach().float().requires_grad_()
        wr = w.detach().float().requires_grad_()
        dxr, dwr = torch.autograd.grad(F.rms_norm(xr, (n,), wr, 1e-6), (xr, wr), g.float())
        try:
            torch.testing.assert_close(dx.float(), dxr, **BWD_TOL[dt])
            torch.testing.assert_close(dw.float(), dwr, **BWD_TOL[dt])
        except AssertionError as e:
            check("bwd numeric", False, f"{m}x{n} {dt}: {str(e).splitlines()[0]}")
        dx2, dw2 = torch.autograd.grad(fl.rms_norm(x, (n,), w, 1e-6), (x, w), g)
        check("bwd determinism", torch.equal(dx, dx2) and torch.equal(dw, dw2),
              f"{m}x{n} {dt}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=40,
                   help="configs for the fused-add and forward passes; "
                        "fp8 runs 3/4 of this, backward 1/2 (default 40)")
    p.add_argument("--seed", type=int, default=20260825)
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable; nothing to fuzz")
        return 0
    torch.manual_seed(args.seed)
    fuzz_fused_add(args.rounds)
    fuzz_forward_numerics(args.rounds)
    fuzz_fp8(max(1, args.rounds * 3 // 4))
    fuzz_backward(max(1, args.rounds // 2))
    print("FUZZ RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
