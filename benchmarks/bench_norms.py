#!/usr/bin/env python3
"""Benchmark the v0.4.0 op family against honest PyTorch competitors.

For each op and shape this reports the same three numbers as
``bench_layernorm.py`` (whose timing machinery it imports): eager per-call
latency, GPU kernel time from ``torch.profiler`` (the only number that may be
quoted as a kernel speedup), and CUDA-graph replay time where capturable.

The competitors are deliberately the strongest available:
  * the eager composite (e.g. ``z = x + r; F.rms_norm(z)``) — what eager
    PyTorch actually runs, AND
  * the same composite under ``torch.compile(fullgraph=True)`` — Inductor
    fuses add+norm, so beating only eager would be a strawman. Compiled
    candidates are warmed before timing; their kernel time is the honest
    comparison (compilation happens once, off the clock).
  * for plain rms_norm: ``F.rms_norm``, which on CUDA dispatches to aten's
    fused ``_fused_rms_norm`` kernel — near-parity is the expected, honest
    outcome there.

The backward rows time a full FORWARD + BACKWARD step per call
(``torch.autograd.grad`` with the forward inside the thunk), ours vs the
composite - label them as training-step numbers, not isolated backward
kernel time.

Per-op ``bytes_moved`` models differ (a fused-add moves ~2x a plain norm), so
GB/s — not raw microseconds — is the honest cross-op metric; each model is
recorded in the JSON.

Usage:
  python benchmarks/bench_norms.py                     # all ops, fp16, default shapes
  python benchmarks/bench_norms.py --op fused_add_rms_norm --dtype float32
  python benchmarks/bench_norms.py --shapes 4096x4096,16384x1024
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_layernorm import (  # noqa: E402
    _fmt,
    _gpu_slug,
    _peak_bandwidth,
    collect_metadata,
    init_profiler_once,
    parse_shapes,
    time_eager_us,
    time_graph_us,
    time_kernel_us,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SHAPES: List[Tuple[int, int]] = [
    (512, 1024),
    (2048, 4096),
    (4096, 4096),
    (8192, 1024),
    (16384, 768),
    (4096, 8192),
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclasses.dataclass
class OpSpec:
    name: str
    # (m, n, dtype, device) -> dict of tensors/state shared by all candidates
    make_inputs: Callable[[int, int, torch.dtype, str], Dict[str, Any]]
    # (inputs) -> [(label, thunk)]; first candidate is "ours"
    candidates: Callable[[Dict[str, Any]], List[Tuple[str, Callable[[], Any]]]]
    # (inputs) -> None, raises AssertionError on mismatch (correctness gate)
    check: Callable[[Dict[str, Any]], None]
    # (m, n, elem_size) -> modelled bytes per call
    bytes_moved: Callable[[int, int, int], int]
    bytes_model: str  # human-readable formula, recorded in the JSON
    graphs: bool = True  # attempt CUDA-graph capture


def _compiled(fn: Callable) -> Callable:
    return torch.compile(fn, fullgraph=True, dynamic=False)


def _base_inputs(m: int, n: int, dtype: torch.dtype, device: str) -> Dict[str, Any]:
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((m, n), generator=g, device=device, dtype=torch.float32).to(dtype)
    r = torch.randn((m, n), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = (torch.rand((n,), generator=g, device=device) + 0.5).to(dtype)
    b = torch.randn((n,), generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x, "r": r, "w": w, "b": b, "n": n, "dtype": dtype}


# --------------------------------------------------------------------------- #
# Op specs
# --------------------------------------------------------------------------- #


def _rms_candidates(i):
    import fused_layernorm

    x, w, n = i["x"], i["w"], i["n"]

    def composite():
        return F.rms_norm(x, (n,), w, 1e-6)

    comp_c = _compiled(composite)
    comp_c()  # warm
    return [
        ("fused_layernorm.rms_norm", lambda: fused_layernorm.rms_norm(x, (n,), w, 1e-6)),
        ("F.rms_norm (aten fused kernel)", composite),
        ("torch.compile(F.rms_norm)", comp_c),
    ]


def _rms_check(i):
    import fused_layernorm

    y = fused_layernorm.rms_norm(i["x"], (i["n"],), i["w"], 1e-6)
    ref = F.rms_norm(i["x"].float(), (i["n"],), i["w"].float(), 1e-6).to(i["dtype"])
    tol = 1e-5 if i["dtype"] == torch.float32 else 2e-2
    assert (y - ref).abs().max().item() < tol


def _fused_add_candidates(rms: bool):
    def make(i):
        import fused_layernorm

        x, r, w, b, n = i["x"], i["r"], i["w"], i["b"], i["n"]
        # The inplace candidate mutates this buffer every call (r += x
        # repeatedly). Reallocating or restoring it inside the thunk would
        # charge the inplace op for work the others do not do, so the drift
        # is accepted; the values it produces are never checked for accuracy.
        r_inplace = r.clone()

        if rms:
            def composite():
                z = x + r
                return F.rms_norm(z, (n,), w, 1e-6), z

            fused = lambda: fused_layernorm.fused_add_rms_norm(x, r, (n,), w, 1e-6)
            fused_inplace = lambda: fused_layernorm.fused_add_rms_norm(
                x, r_inplace, (n,), w, 1e-6, inplace=True
            )
        else:
            def composite():
                z = x + r
                return F.layer_norm(z, (n,), w, b, 1e-5), z

            fused = lambda: fused_layernorm.fused_add_layer_norm(x, r, (n,), w, b, 1e-5)
            fused_inplace = lambda: fused_layernorm.fused_add_layer_norm(
                x, r_inplace, (n,), w, b, 1e-5, inplace=True
            )

        comp_c = _compiled(composite)
        comp_c()  # warm
        return [
            ("fused op", fused),
            ("fused op (inplace)", fused_inplace),
            ("eager composite (add; norm)", composite),
            ("torch.compile(composite)", comp_c),
        ]

    return make


def _fused_add_check(rms: bool):
    def check(i):
        import fused_layernorm

        x, r, w, b, n = i["x"], i["r"], i["w"], i["b"], i["n"]
        if rms:
            out, z = fused_layernorm.fused_add_rms_norm(x, r, (n,), w, 1e-6)
            assert torch.equal(z, x + r)
            assert torch.equal(out, fused_layernorm.rms_norm(z, (n,), w, 1e-6))
        else:
            out, z = fused_layernorm.fused_add_layer_norm(x, r, (n,), w, b, 1e-5)
            assert torch.equal(z, x + r)
            assert torch.equal(out, fused_layernorm.layer_norm(z, (n,), w, b, 1e-5))

    return check


def _fp8_dynamic_candidates(i):
    import fused_layernorm

    x, w, n = i["x"], i["w"], i["n"]

    def composite():
        y = F.rms_norm(x, (n,), w, 1e-6)
        yf = y.float()  # hoisted: computing it twice would handicap the competitor
        s = (yf.abs().amax(-1, keepdim=True).clamp(min=1e-12)) / 448.0
        q = (yf * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q, s

    comp_c = _compiled(composite)
    comp_c()
    return [
        ("fused rms_norm_fp8 (dynamic)",
         lambda: fused_layernorm.rms_norm_fp8(x, (n,), w, 1e-6)),
        ("eager composite (norm; amax; quant)", composite),
        ("torch.compile(composite)", comp_c),
    ]


def _fp8_dynamic_check(i):
    import fused_layernorm

    x, w, n = i["x"], i["w"], i["n"]
    out, s = fused_layernorm.rms_norm_fp8(x, (n,), w, 1e-6)
    y = fused_layernorm.rms_norm(x, (n,), w, 1e-6)
    # Independent scale check first (quantising with the kernel's own scale
    # alone would let a wrong-but-consistent scale pass). Relative bound: the
    # kernel's fp32 amax/448 and this one can differ by an ulp.
    s_ref = y.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    assert ((s - s_ref).abs() <= s_ref * 1e-6 + 1e-12).all()
    q = (y.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), q.view(torch.uint8))


def _fused_add_fp8_candidates(i):
    import fused_layernorm

    x, r, w, n = i["x"], i["r"], i["w"], i["n"]

    def composite():
        z = x + r
        y = F.rms_norm(z, (n,), w, 1e-6)
        yf = y.float()  # hoisted: see the plain fp8 composite
        s = (yf.abs().amax(-1, keepdim=True).clamp(min=1e-12)) / 448.0
        q = (yf * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q, z, s

    comp_c = _compiled(composite)
    comp_c()
    return [
        ("fused_add_rms_norm_fp8 (dynamic)",
         lambda: fused_layernorm.fused_add_rms_norm_fp8(x, r, (n,), w, 1e-6)),
        ("eager composite", composite),
        ("torch.compile(composite)", comp_c),
    ]


def _fused_add_fp8_check(i):
    import fused_layernorm

    x, r, w, n = i["x"], i["r"], i["w"], i["n"]
    out, z, s = fused_layernorm.fused_add_rms_norm_fp8(x, r, (n,), w, 1e-6)
    assert torch.equal(z, x + r)
    y = fused_layernorm.rms_norm(z, (n,), w, 1e-6)
    q = (y.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), q.view(torch.uint8))


def _bwd_inputs(m: int, n: int, dtype: torch.dtype, device: str) -> Dict[str, Any]:
    i = _base_inputs(m, n, dtype, device)
    i["gy"] = torch.randn_like(i["x"])
    return i


def _bwd_candidates(rms: bool):
    def make(i):
        import fused_layernorm

        x, w, b, n, gy = i["x"], i["w"], i["b"], i["n"], i["gy"]

        def ours():
            xg = x.detach().requires_grad_()
            wg = w.detach().requires_grad_()
            if rms:
                y = fused_layernorm.rms_norm(xg, (n,), wg, 1e-6)
                return torch.autograd.grad(y, (xg, wg), gy)
            bg = b.detach().requires_grad_()
            y = fused_layernorm.layer_norm(xg, (n,), wg, bg, 1e-5)
            return torch.autograd.grad(y, (xg, wg, bg), gy)

        def composite():
            xg = x.detach().requires_grad_()
            wg = w.detach().requires_grad_()
            if rms:
                y = F.rms_norm(xg, (n,), wg, 1e-6)
                return torch.autograd.grad(y, (xg, wg), gy)
            bg = b.detach().requires_grad_()
            y = F.layer_norm(xg, (n,), wg, bg, 1e-5)
            return torch.autograd.grad(y, (xg, wg, bg), gy)

        return [
            ("ours: fwd_train + CUDA bwd", ours),
            ("PyTorch autograd composite", composite),
        ]

    return make


def _bwd_check(rms: bool):
    def check(i):
        import fused_layernorm

        x, w, n, gy = i["x"], i["w"], i["n"], i["gy"]
        xg = x.detach().requires_grad_()
        wg = w.detach().requires_grad_()
        xc = x.detach().requires_grad_()
        wc = w.detach().requires_grad_()
        if rms:
            dy = torch.autograd.grad(fused_layernorm.rms_norm(xg, (n,), wg, 1e-6), (xg, wg), gy)
            dc = torch.autograd.grad(F.rms_norm(xc, (n,), wc, 1e-6), (xc, wc), gy)
        else:
            b = i["b"]
            bg = b.detach().requires_grad_()
            bc = b.detach().requires_grad_()
            dy = torch.autograd.grad(
                fused_layernorm.layer_norm(xg, (n,), wg, bg, 1e-5), (xg, wg, bg), gy
            )
            dc = torch.autograd.grad(F.layer_norm(xc, (n,), wc, bc, 1e-5), (xc, wc, bc), gy)
        tol = 1e-4 if i["dtype"] == torch.float32 else 5e-2
        for a, c in zip(dy, dc):
            assert (a - c).abs().max().item() < tol * max(1.0, c.abs().max().item())

    return check


def _ln_fp8_dynamic_candidates(i):
    import fused_layernorm

    x, w, b, n = i["x"], i["w"], i["b"], i["n"]

    def composite():
        y = F.layer_norm(x, (n,), w, b, 1e-5)
        yf = y.float()  # hoisted: see the RMS fp8 composite
        s = (yf.abs().amax(-1, keepdim=True).clamp(min=1e-12)) / 448.0
        q = (yf * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q, s

    comp_c = _compiled(composite)
    comp_c()
    return [
        ("fused layer_norm_fp8 (dynamic)",
         lambda: fused_layernorm.layer_norm_fp8(x, (n,), w, b, 1e-5)),
        ("eager composite (norm; amax; quant)", composite),
        ("torch.compile(composite)", comp_c),
    ]


def _ln_fp8_dynamic_check(i):
    import fused_layernorm

    x, w, b, n = i["x"], i["w"], i["b"], i["n"]
    out, s = fused_layernorm.layer_norm_fp8(x, (n,), w, b, 1e-5)
    y = fused_layernorm.layer_norm(x, (n,), w, b, 1e-5)
    s_ref = y.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    assert ((s - s_ref).abs() <= s_ref * 1e-6 + 1e-12).all()
    q = (y.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), q.view(torch.uint8))


def _ln_fused_add_fp8_candidates(i):
    import fused_layernorm

    x, r, w, b, n = i["x"], i["r"], i["w"], i["b"], i["n"]

    def composite():
        z = x + r
        y = F.layer_norm(z, (n,), w, b, 1e-5)
        yf = y.float()  # hoisted: see the plain fp8 composite
        s = (yf.abs().amax(-1, keepdim=True).clamp(min=1e-12)) / 448.0
        q = (yf * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q, z, s

    comp_c = _compiled(composite)
    comp_c()
    return [
        ("fused_add_layer_norm_fp8 (dynamic)",
         lambda: fused_layernorm.fused_add_layer_norm_fp8(x, r, (n,), w, b, 1e-5)),
        ("eager composite", composite),
        ("torch.compile(composite)", comp_c),
    ]


def _ln_fused_add_fp8_check(i):
    import fused_layernorm

    x, r, w, b, n = i["x"], i["r"], i["w"], i["b"], i["n"]
    out, z, s = fused_layernorm.fused_add_layer_norm_fp8(x, r, (n,), w, b, 1e-5)
    assert torch.equal(z, x + r)
    y = fused_layernorm.layer_norm(z, (n,), w, b, 1e-5)
    s_ref = y.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    assert ((s - s_ref).abs() <= s_ref * 1e-6 + 1e-12).all()
    q = (y.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), q.view(torch.uint8))


def _gelu_bwd_candidates(i):
    import fused_layernorm

    x, w, b, n, gy = i["x"], i["w"], i["b"], i["n"], i["gy"]

    def ours():
        xg = x.detach().requires_grad_()
        wg = w.detach().requires_grad_()
        bg = b.detach().requires_grad_()
        y = fused_layernorm.layer_norm_gelu(xg, (n,), wg, bg, 1e-5)
        return torch.autograd.grad(y, (xg, wg, bg), gy)

    def composite():
        xg = x.detach().requires_grad_()
        wg = w.detach().requires_grad_()
        bg = b.detach().requires_grad_()
        y = F.gelu(F.layer_norm(xg, (n,), wg, bg, 1e-5))
        return torch.autograd.grad(y, (xg, wg, bg), gy)

    return [
        ("ours: fwd_train + CUDA bwd", ours),
        ("PyTorch autograd composite", composite),
    ]


def _gelu_bwd_check(i):
    import fused_layernorm

    x, w, b, n, gy = i["x"], i["w"], i["b"], i["n"], i["gy"]
    xg = x.detach().requires_grad_()
    wg = w.detach().requires_grad_()
    bg = b.detach().requires_grad_()
    xc = x.detach().requires_grad_()
    wc = w.detach().requires_grad_()
    bc = b.detach().requires_grad_()
    dy = torch.autograd.grad(
        fused_layernorm.layer_norm_gelu(xg, (n,), wg, bg, 1e-5), (xg, wg, bg), gy
    )
    dc = torch.autograd.grad(
        F.gelu(F.layer_norm(xc, (n,), wc, bc, 1e-5)), (xc, wc, bc), gy
    )
    tol = 1e-4 if i["dtype"] == torch.float32 else 5e-2
    for a, c in zip(dy, dc):
        assert (a - c).abs().max().item() < tol * max(1.0, c.abs().max().item())


OPS: Dict[str, OpSpec] = {
    "rms_norm": OpSpec(
        name="rms_norm",
        make_inputs=_base_inputs,
        candidates=_rms_candidates,
        check=_rms_check,
        bytes_moved=lambda m, n, e: 2 * m * n * e + n * e,
        bytes_model="2*M*N*e + N*e (read x, write y, read weight)",
    ),
    "fused_add_layer_norm": OpSpec(
        name="fused_add_layer_norm",
        make_inputs=_base_inputs,
        candidates=_fused_add_candidates(rms=False),
        check=_fused_add_check(rms=False),
        bytes_moved=lambda m, n, e: 4 * m * n * e + 2 * n * e,
        bytes_model="4*M*N*e + 2*N*e (read x, read r, write z, write y, params)",
    ),
    "fused_add_rms_norm": OpSpec(
        name="fused_add_rms_norm",
        make_inputs=_base_inputs,
        candidates=_fused_add_candidates(rms=True),
        check=_fused_add_check(rms=True),
        bytes_moved=lambda m, n, e: 4 * m * n * e + n * e,
        bytes_model="4*M*N*e + N*e (read x, read r, write z, write y, weight)",
    ),
    "rms_norm_fp8_dynamic": OpSpec(
        name="rms_norm_fp8_dynamic",
        make_inputs=_base_inputs,
        candidates=_fp8_dynamic_candidates,
        check=_fp8_dynamic_check,
        bytes_moved=lambda m, n, e: m * n * e + m * n * 1 + n * e + 4 * m,
        bytes_model="M*N*e + M*N*1 + N*e + 4*M (read x, write fp8, weight, scales)",
    ),
    "fused_add_rms_norm_fp8_dynamic": OpSpec(
        name="fused_add_rms_norm_fp8_dynamic",
        make_inputs=_base_inputs,
        candidates=_fused_add_fp8_candidates,
        check=_fused_add_fp8_check,
        bytes_moved=lambda m, n, e: 3 * m * n * e + m * n * 1 + n * e + 4 * m,
        bytes_model="3*M*N*e + M*N*1 + N*e + 4*M (read x, read r, write z, write fp8, ...)",
    ),
    "layer_norm_fp8_dynamic": OpSpec(
        name="layer_norm_fp8_dynamic",
        make_inputs=_base_inputs,
        candidates=_ln_fp8_dynamic_candidates,
        check=_ln_fp8_dynamic_check,
        bytes_moved=lambda m, n, e: m * n * e + m * n * 1 + 2 * n * e + 4 * m,
        bytes_model="M*N*e + M*N*1 + 2*N*e + 4*M (read x, write fp8, params, scales)",
    ),
    "fused_add_layer_norm_fp8_dynamic": OpSpec(
        name="fused_add_layer_norm_fp8_dynamic",
        make_inputs=_base_inputs,
        candidates=_ln_fused_add_fp8_candidates,
        check=_ln_fused_add_fp8_check,
        bytes_moved=lambda m, n, e: 3 * m * n * e + m * n * 1 + 2 * n * e + 4 * m,
        bytes_model="3*M*N*e + M*N*1 + 2*N*e + 4*M (read x, read r, write z, write fp8, ...)",
    ),
    "layer_norm_gelu_bwd": OpSpec(
        name="layer_norm_gelu_bwd",
        make_inputs=_bwd_inputs,
        candidates=_gelu_bwd_candidates,
        check=_gelu_bwd_check,
        bytes_moved=lambda m, n, e: 3 * m * n * e + 2 * n * e,
        bytes_model="~3*M*N*e (read dy, read x, write dx) + param grads",
        graphs=False,
    ),
    "layer_norm_bwd": OpSpec(
        name="layer_norm_bwd",
        make_inputs=_bwd_inputs,
        candidates=_bwd_candidates(rms=False),
        check=_bwd_check(rms=False),
        bytes_moved=lambda m, n, e: 3 * m * n * e + 2 * n * e,
        bytes_model="~3*M*N*e (read dy, read x, write dx) + param grads",
        graphs=False,  # autograd engine paths are not capture-friendly here
    ),
    "rms_norm_bwd": OpSpec(
        name="rms_norm_bwd",
        make_inputs=_bwd_inputs,
        candidates=_bwd_candidates(rms=True),
        check=_bwd_check(rms=True),
        bytes_moved=lambda m, n, e: 3 * m * n * e + n * e,
        bytes_model="~3*M*N*e (read dy, read x, write dx) + param grads",
        graphs=False,
    ),
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def bench_op(
    spec: OpSpec,
    shapes: Sequence[Tuple[int, int]],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    device = "cuda"
    results = []
    for m, n in shapes:
        print(f"- {spec.name} {m}x{n} {dtype} ...", flush=True)
        inputs = spec.make_inputs(m, n, dtype, device)
        entry: Dict[str, Any] = {
            "op": spec.name,
            "M": m,
            "N": n,
            "dtype": str(dtype).replace("torch.", ""),
            "bytes_moved": spec.bytes_moved(m, n, inputs["x"].element_size()),
            "bytes_model": spec.bytes_model,
            "candidates": {},
        }
        try:
            spec.check(inputs)
            entry["correctness"] = "passed"
        except AssertionError as exc:
            entry["correctness"] = f"FAILED: {exc}"
            results.append(entry)
            continue

        cands = spec.candidates(inputs)
        eager = time_eager_us(cands, iters=args.iters, reps=args.reps, warmup=args.warmup)
        for label, fn in cands:
            eager_res = eager.get(label, {})
            c: Dict[str, Any] = dict(eager_res)
            try:
                c.update(time_kernel_us(fn, iters=args.iters, warmup=args.warmup))
            except Exception as exc:  # profiler failure: record, don't die
                c["kernel_error"] = repr(exc)
            if spec.graphs:
                c.update(time_graph_us(fn, iters=args.iters, reps=args.reps))
            entry["candidates"][label] = c
        results.append(entry)
    return results


def render(results: List[Dict[str, Any]], peak: Optional[float]) -> str:
    lines = [
        "| op | MxN | dtype | candidate | eager us | kernel us | GB/s | % peak |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        for label, c in r.get("candidates", {}).items():
            k = c.get("kernel_us")
            gbs = (r["bytes_moved"] / (k * 1e-6) / 1e9) if k else None
            pct = (100 * gbs / peak) if (gbs and peak) else None
            lines.append(
                f"| {r['op']} | {r['M']}x{r['N']} | {r['dtype']} | {label} | "
                f"{_fmt(c.get('eager_us'))} | {_fmt(k)} | {_fmt(gbs, 1)} | {_fmt(pct, 1)} |"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--op", action="append", choices=sorted(OPS), default=None,
                   help="repeatable; default: all ops")
    p.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    p.add_argument("--shapes", type=parse_shapes, default=DEFAULT_SHAPES)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "out"))
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA GPU required", file=sys.stderr)
        return 1
    try:
        import fused_layernorm_cuda as ext  # noqa: F401
    except ImportError:
        print("fused_layernorm_cuda not built", file=sys.stderr)
        return 1
    import fused_layernorm

    if not fused_layernorm.is_available():
        print("fused_layernorm reports unavailable", file=sys.stderr)
        return 1

    init_profiler_once()
    meta = collect_metadata(REPO_ROOT, ext)
    meta["script"] = "bench_norms.py"
    meta["argv"] = sys.argv[1:]
    dtype = DTYPES[args.dtype]
    ops = args.op or sorted(OPS)

    all_results: List[Dict[str, Any]] = []
    for name in ops:
        all_results.extend(bench_op(OPS[name], args.shapes, dtype, args))

    peak = _peak_bandwidth(meta["gpu_name"])
    table = render(all_results, peak)
    print()
    print(table)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = out_dir / f"bench_norms_{stamp}_{_gpu_slug(meta['gpu_name'])}_{args.dtype}"
    base.with_suffix(".json").write_text(
        json.dumps({"metadata": meta, "results": all_results}, indent=2)
    )
    base.with_suffix(".md").write_text(
        f"# bench_norms ({meta['timestamp']})\n\n"
        f"{meta['gpu_name']}, torch {meta['torch_version']}, CUDA {meta['cuda_version']}, "
        f"commit {meta['git_commit']} (dirty={meta['git_dirty']}), "
        f"driver {meta.get('nvidia_driver')}.\n\n" + table + "\n"
    )
    print(f"\nwrote {base}.json\nwrote {base}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
