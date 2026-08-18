#!/usr/bin/env python3
"""Benchmark ``fused_layernorm_cuda.layernorm`` against PyTorch's LayerNorm.

Dependencies: ``torch`` and the Python standard library only.  Requires a CUDA
GPU and the compiled ``fused_layernorm_cuda`` extension; without them the script
prints how to obtain them and exits with status 1.

For every shape (M rows x N columns) and every candidate implementation the
script reports THREE different numbers.  They answer different questions and
must not be mixed up:

(1) ``eager_us`` - eager per-call latency, in microseconds per call.
    ``time.perf_counter()`` around a loop of ``--iters`` calls followed by
    ``torch.cuda.synchronize()``, repeated ``--reps`` times; the median is
    reported.  The repetitions are interleaved round-robin across candidates
    (after warming all of them up) so that drift affects them alike.  This
    INCLUDES Python, dispatcher and kernel-launch overhead.  At
    small sizes (a few KB to a few MB) that overhead dominates and the number is
    almost independent of the tensor size.  This is what the legacy scripts in
    ``benchmarks/legacy/`` measured (with additional flaws, see the README
    there).  It tells you what a user pays per Python call, not how fast the
    kernel is.

(2) ``kernel_us`` - GPU kernel time, in microseconds per call, from
    ``torch.profiler`` (CUDA activity only): the sum of all device-side kernel
    durations over ``--iters`` calls divided by ``--iters``.  The kernel names
    are recorded too, so a reader can see e.g. ``vectorized_layer_norm_kernel``
    versus ``RowwiseMomentsCUDAKernel`` + ``LayerNormForwardCUDAKernel`` for
    PyTorch, and ``layernorm_kernel`` for this extension.  THIS is the only
    number that says anything about the kernel itself, and the only one that
    may be quoted as a "kernel speedup".

(3) ``graph_us`` - CUDA-graph replay time, in microseconds per call: ``--iters``
    calls are captured into one ``torch.cuda.CUDAGraph`` and the replay is
    timed with CUDA events.  This removes Python/dispatch cost and shows what
    the op costs when it is launched from a graph; it also documents whether a
    candidate is capturable at all (a kernel launched on the legacy default
    stream cannot be captured; capture errors are recorded, not hidden).

Effective bandwidth (GB/s) and utilisation of the device's peak bandwidth are
derived from ``kernel_us`` only, using bytes = M*N*elem*2 + N*elem*2 (read x,
write y, read weight and bias).  Peak bandwidth is looked up from a small table
keyed on ``torch.cuda.get_device_name()``; for unknown devices utilisation is
not computed.

Before timing, every shape is checked for correctness against
``torch.nn.functional.layer_norm`` with random (non-identity) weight and bias;
a shape that fails is reported and skipped.

Outputs: a JSON file and a Markdown table in ``--out`` (default
``benchmarks/out/``, which is git-ignored).  Nothing in this file contains a
measured number; all numbers come from running it.

Limitations: forward pass only (the extension has no backward); one GPU;
timings depend on GPU clocks, driver, and host CPU - always commit the JSON
together with the environment metadata it contains.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_SHAPES: List[Tuple[int, int]] = [
    (32, 768),
    (32, 1024),
    (32, 4096),
    (17, 1023),
    (128, 4096),
    (512, 1024),
    (2048, 4096),
    (4096, 4096),
    (8192, 1024),
    (16384, 768),
    (4096, 12288),
]

DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}

# Correctness tolerances used for the pre-check (ours vs F.layer_norm in the same
# dtype).  The fp32 pair matches tests/test_layernorm.py; the others are loose
# because PyTorch and this kernel may round differently in low precision.
CHECK_TOL: Dict[str, Tuple[float, float]] = {
    "float32": (1e-5, 1e-4),
    "float64": (1e-10, 1e-8),
    "float16": (1e-2, 1e-2),
    "bfloat16": (5e-2, 5e-2),
}

# Peak HBM bandwidth in GB/s keyed on substrings of torch.cuda.get_device_name().
# Sources: NVIDIA A100 datasheet (40 GB: 1555, 80 GB PCIe: 1935, 80 GB SXM: 2039),
# NVIDIA H100 datasheet (SXM HBM3: 3350, PCIe: 2000), NVIDIA V100 datasheet (900).
# Substrings are matched in order; anything not listed gets "peak unknown".
PEAK_BANDWIDTH_GB_S: List[Tuple[str, float]] = [
    ("A100-SXM4-80GB", 2039.0),
    ("A100-SXM4-40GB", 1555.0),
    ("A100-PCIE-40GB", 1555.0),
    ("A100 80GB PCIe", 1935.0),
    ("H100 80GB HBM3", 3350.0),
    ("H100 PCIe", 2000.0),
    ("V100", 900.0),
]

BUILD_HELP = """\
The compiled extension `fused_layernorm_cuda` is not importable.
Build it in the repository root on a machine with CUDA and a GPU:

    pip install --no-build-isolation -e .   # or: python setup.py build_ext --inplace

then re-run:

    python benchmarks/bench_layernorm.py
"""


# --------------------------------------------------------------------------- #
# Environment / metadata helpers
# --------------------------------------------------------------------------- #

def _git(repo_root: Path, *cmd: str) -> Optional[str]:
    """stdout of ``git <cmd>`` in ``repo_root``, or None if git is unavailable/fails."""
    try:
        out = subprocess.run(
            ["git", *cmd],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout
    except Exception:  # git missing, not a repo, ...
        pass
    return None


def _git_commit(repo_root: Path) -> Optional[str]:
    out = _git(repo_root, "rev-parse", "HEAD")
    return out.strip() if out is not None else None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    """True if the working tree has uncommitted changes, None if unknown."""
    out = _git(repo_root, "status", "--porcelain")
    return bool(out.strip()) if out is not None else None


def _cpu_model() -> str:
    name = platform.processor() or ""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        return f"{model} ({name})" if name and name != model else model
        except OSError:
            pass
    return name or platform.machine() or "unknown"


def _peak_bandwidth(device_name: str) -> Optional[float]:
    for key, peak in PEAK_BANDWIDTH_GB_S:
        if key in device_name:
            return peak
    return None


def _gpu_slug(device_name: str) -> str:
    slug = device_name.lower().replace("nvidia", "").strip()
    return "".join(ch if ch.isalnum() else "-" for ch in slug).strip("-") or "gpu"


def collect_metadata(repo_root: Path, ext_module: Any) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_name = torch.cuda.get_device_name()
    cudnn_version: Optional[int]
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu_name": device_name,
        "gpu_sm_count": props.multi_processor_count,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_peak_bandwidth_gb_s": _peak_bandwidth(device_name),
        "cpu_model": _cpu_model(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "extension_version": getattr(ext_module, "__version__", None),
        "argv": sys.argv[1:],
    }


# --------------------------------------------------------------------------- #
# The three measurements
# --------------------------------------------------------------------------- #

def _eager_rep_us(fn: Callable[[], Any], iters: int) -> float:
    """One timed repetition: perf_counter around ``iters`` calls + synchronize; us/call."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters


def time_eager_us(
    candidates: Sequence[Tuple[str, Callable[[], Any]]],
    iters: int,
    reps: int,
    warmup: int,
) -> Dict[str, Dict[str, Any]]:
    """(1) Eager per-call latency for every candidate, measured round-robin.

    All candidates are warmed up first; then each of the ``reps`` repetitions
    times ``iters`` calls of candidate 1, then of candidate 2, ... so that slow
    drifts (clocks, thermal state, background load) affect all candidates alike
    instead of whichever happened to be measured last.  Returns, per candidate
    name, either the timing dict or ``{"error": str}``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    per_call: Dict[str, List[float]] = {}
    failed: Dict[str, str] = {}
    for name, fn in candidates:
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            per_call[name] = []
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
    for _ in range(reps):
        for name, fn in candidates:
            if name in failed:
                continue
            try:
                per_call[name].append(_eager_rep_us(fn, iters))
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {exc}"
    for name, _ in candidates:
        if name in failed:
            results[name] = {"error": failed[name]}
            continue
        vals = per_call[name]
        results[name] = {
            "eager_us": statistics.median(vals),
            "eager_us_min": min(vals),
            "eager_us_max": max(vals),
            "eager_us_reps": vals,
        }
    return results


def _event_device_us(evt: Any) -> float:
    """Device time (us) of one profiler event, robust across torch versions.

    Newer torch exposes ``device_time``; older versions only ``cuda_time``.
    """
    if hasattr(evt, "device_time"):
        val = evt.device_time
    else:
        val = getattr(evt, "cuda_time", 0)
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_device_event(evt: Any) -> bool:
    dev = getattr(evt, "device_type", None)
    if dev is None:
        return False
    return "CUDA" in str(dev).upper()


# Kineto reports GPU memcpy/memset and (when CUDA-sync tracing is enabled) host
# synchronisation events with the same device type as kernels.  They are not
# kernel time, so they are excluded from ``kernel_us`` and counted separately.
_NON_KERNEL_PREFIXES = ("memcpy", "memset")
_NON_KERNEL_MARKERS = ("context sync", "stream sync", "event sync", "stream wait event",
                       "cudadevicesynchronize", "cudastreamsynchronize", "cudaeventsynchronize")


def _is_kernel_event(evt: Any) -> bool:
    if not _is_device_event(evt):
        return False
    name = str(getattr(evt, "name", "")).strip().lower()
    if name.startswith(_NON_KERNEL_PREFIXES):
        return False
    return not any(marker in name for marker in _NON_KERNEL_MARKERS)


def _short_kernel_name(name: str) -> str:
    """'void at::native::(anonymous namespace)::foo<float>(...)' -> 'foo'."""
    n = name.strip().replace("(anonymous namespace)", "")
    if n.startswith("void "):
        n = n[5:]
    for cut in ("(", "<"):
        idx = n.find(cut)
        if idx > 0:
            n = n[:idx]
    if "::" in n:
        n = n.rsplit("::", 1)[-1]
    return n.strip() or name


def init_profiler_once() -> None:
    """Run one throwaway CUDA profiler session before any CUDA-graph capture.

    Initialising CUPTI for the first time *after* a graph has been captured is
    known to be problematic on older CUDA versions (see torch.profiler's own
    ``_utils._init_for_cuda_graphs``), so the profiler is initialised up front
    in ``main()`` rather than relying on the order of measurements.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with profile(activities=[ProfilerActivity.CUDA]):
                torch.cuda.synchronize()
    except Exception:  # profiler unavailable: time_kernel_us will report the error per candidate
        pass


def time_kernel_us(fn: Callable[[], Any], iters: int, warmup: int) -> Dict[str, Any]:
    """(2) GPU kernel time per call from torch.profiler (CUDA activity only)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.profiler emits informational warnings
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()

    total_us = 0.0
    non_kernel_events = 0
    names: Dict[str, int] = {}
    full_names: Dict[str, int] = {}
    for evt in prof.events():
        if not _is_device_event(evt):
            continue
        if not _is_kernel_event(evt):
            non_kernel_events += 1  # memcpy/memset/sync: device-side but not kernel time
            continue
        us = _event_device_us(evt)
        if us <= 0:
            continue
        total_us += us
        short = _short_kernel_name(evt.name)
        names[short] = names.get(short, 0) + 1
        full_names[evt.name] = full_names.get(evt.name, 0) + 1
    return {
        "kernel_us": total_us / iters if total_us > 0 else None,
        "kernel_launches_per_call": (sum(names.values()) / iters) if names else 0.0,
        "kernel_names": names,  # short name -> number of device events over `iters` calls
        "kernel_names_per_call": {k: v / iters for k, v in names.items()},
        "kernel_names_full": full_names,
        "non_kernel_device_events": non_kernel_events,
    }


def time_graph_us(fn: Callable[[], Any], iters: int, reps: int) -> Dict[str, Any]:
    """(3) CUDA-graph replay time per call; records the error if capture fails."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()

        graph.replay()  # warm replay
        torch.cuda.synchronize()
        per_call: List[float] = []
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            per_call.append(start.elapsed_time(end) * 1e3 / iters)
        return {
            "graph_us": statistics.median(per_call),
            "graph_us_min": min(per_call),
            "graph_us_max": max(per_call),
            "graph_error": None,
        }
    except Exception as exc:  # capture not supported for this candidate
        return {"graph_us": None, "graph_error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Per-shape driver
# --------------------------------------------------------------------------- #

def bytes_moved(m: int, n: int, elem_size: int) -> int:
    return m * n * elem_size * 2 + n * elem_size * 2


def make_candidates(
    ext_module: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> List[Tuple[str, Callable[[], torch.Tensor]]]:
    n = x.shape[-1]
    module = torch.nn.LayerNorm(n, eps=eps, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
    return [
        ("nn.LayerNorm (module)", lambda: module(x)),
        ("F.layer_norm", lambda: torch.nn.functional.layer_norm(x, (n,), weight, bias, eps)),
        ("fused_layernorm_cuda.layernorm", lambda: ext_module.layernorm(x, weight, bias, eps)),
    ]


def correctness_check(
    ext_module: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dtype_name: str,
) -> Tuple[bool, str]:
    atol, rtol = CHECK_TOL[dtype_name]
    ref = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    out = ext_module.layernorm(x, weight, bias, eps)
    torch.cuda.synchronize()
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return False, f"shape/dtype mismatch: got {tuple(out.shape)} {out.dtype}, want {tuple(ref.shape)} {ref.dtype}"
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    max_abs = (out.float() - ref.float()).abs().max().item()
    msg = f"max_abs_diff={max_abs:.3e} (atol={atol}, rtol={rtol})"
    return bool(ok), msg


def bench_shape(
    ext_module: Any,
    m: int,
    n: int,
    dtype_name: str,
    args: argparse.Namespace,
    peak_gb_s: Optional[float],
) -> Dict[str, Any]:
    dtype = DTYPES[dtype_name]
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, device=device, dtype=torch.float32, generator=gen).to(dtype)
    weight = torch.empty(n, device=device, dtype=torch.float32).uniform_(0.5, 1.5, generator=gen).to(dtype)
    bias = torch.randn(n, device=device, dtype=torch.float32, generator=gen).to(dtype)
    elem = x.element_size()

    record: Dict[str, Any] = {
        "M": m,
        "N": n,
        "dtype": dtype_name,
        "bytes_moved": bytes_moved(m, n, elem),
        "correctness": None,
        "candidates": {},
    }

    try:
        ok, msg = correctness_check(ext_module, x, weight, bias, args.eps, dtype_name)
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    record["correctness"] = {"passed": ok, "detail": msg}
    if not ok:
        print(f"  [{m}x{n} {dtype_name}] correctness check FAILED ({msg}); shape skipped.")
        return record

    candidates = make_candidates(ext_module, x, weight, bias, args.eps)
    # (1) eager latency, round-robin over candidates (see time_eager_us).
    eager = time_eager_us(candidates, args.iters, args.reps, args.warmup)
    for name, fn in candidates:
        entry: Dict[str, Any] = {"error": None}
        entry.update(eager[name])
        if entry.get("error"):
            if args.verbose:
                print(f"    {name}: {entry['error']}")
            entry.update({"bandwidth_gb_s": None, "utilization_pct": None})
            record["candidates"][name] = entry
            continue
        try:
            # (2) kernel time, then (3) graph replay, per candidate.
            entry.update(time_kernel_us(fn, args.iters, args.warmup))
            if args.no_graphs:
                entry.update({"graph_us": None, "graph_error": "disabled by --no-graphs"})
            else:
                entry.update(time_graph_us(fn, args.iters, args.reps))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if args.verbose:
                traceback.print_exc()
        kernel_us = entry.get("kernel_us")
        if kernel_us:
            gb_s = record["bytes_moved"] / (kernel_us * 1e-6) / 1e9
            entry["bandwidth_gb_s"] = gb_s
            entry["utilization_pct"] = (gb_s / peak_gb_s * 100.0) if peak_gb_s else None
        else:
            entry["bandwidth_gb_s"] = None
            entry["utilization_pct"] = None
        record["candidates"][name] = entry
        torch.cuda.synchronize()
    return record


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(v: Optional[float], nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_table(results: List[Dict[str, Any]], peak_gb_s: Optional[float]) -> str:
    header = [
        "shape (MxN)", "dtype", "candidate", "eager us/call", "kernel us/call",
        "graph us/call", "GB/s (kernel)", "% peak", "kernels seen",
    ]
    rows: List[List[str]] = []
    for rec in results:
        shape = f"{rec['M']}x{rec['N']}"
        if not rec["correctness"]["passed"]:
            rows.append([shape, rec["dtype"], "(skipped: correctness check failed)",
                         "-", "-", "-", "-", "-", rec["correctness"]["detail"]])
            continue
        for cand, e in rec["candidates"].items():
            if e.get("error"):
                rows.append([shape, rec["dtype"], cand, "error", "error", "error", "-", "-", e["error"]])
                continue
            graph = _fmt(e.get("graph_us")) if e.get("graph_us") is not None else (
                "not capturable" if e.get("graph_error") and "disabled" not in e["graph_error"] else "n/a")
            per_call = e.get("kernel_names_per_call") or {}
            kernels = ", ".join(f"{k} x{v:g}/call" for k, v in per_call.items()) or "-"
            rows.append([
                shape, rec["dtype"], cand,
                _fmt(e.get("eager_us")), _fmt(e.get("kernel_us")), graph,
                _fmt(e.get("bandwidth_gb_s"), 1),
                _fmt(e.get("utilization_pct"), 1) if peak_gb_s else "peak unknown",
                kernels,
            ])
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [line, sep]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    if not peak_gb_s:
        out.append("")
        out.append("peak unknown; utilization not computed")
    return "\n".join(out)


def render_markdown(meta: Dict[str, Any], table: str, args: argparse.Namespace) -> str:
    lines = [
        "# bench_layernorm.py results",
        "",
        f"- timestamp: {meta['timestamp']}",
        f"- command: `{' '.join(['python benchmarks/bench_layernorm.py', *meta['argv']]).strip()}`",
        f"- git commit: {meta['git_commit']}"
        + (" (working tree dirty)" if meta.get('git_dirty') else ""),
        f"- torch {meta['torch_version']}, CUDA {meta['cuda_version']}, cuDNN {meta['cudnn_version']}",
        f"- GPU: {meta['gpu_name']} ({meta['gpu_sm_count']} SMs), peak bandwidth used: "
        f"{meta['gpu_peak_bandwidth_gb_s'] or 'unknown'} GB/s",
        f"- CPU: {meta['cpu_model']}; Python {meta['python_version']}",
        f"- extension version: {meta['extension_version']}",
        f"- iters={args.iters}, reps={args.reps}, warmup={args.warmup}, eps={args.eps}, dtype={args.dtype}",
        "",
        "Columns: `eager us/call` = Python-level latency incl. dispatch/launch (median of reps); "
        "`kernel us/call` = summed GPU kernel time from torch.profiler / iters "
        "(the only column that describes the kernel); `graph us/call` = CUDA-graph replay time / iters; "
        "`GB/s` and `% peak` are computed from `kernel us/call`.",
        "",
        table,
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_shapes(text: str) -> List[Tuple[int, int]]:
    shapes: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            m_str, n_str = item.replace("×", "x").split("x")
            m, n = int(m_str), int(n_str)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad shape {item!r}; expected MxN like 32x768") from None
        if m < 0 or n < 0:
            raise argparse.ArgumentTypeError(f"bad shape {item!r}; M and N must be non-negative")
        shapes.append((m, n))
    if not shapes:
        raise argparse.ArgumentTypeError("no shapes given")
    return shapes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python benchmarks/bench_layernorm.py --shapes 32x768,4096x4096 --dtype float16",
    )
    p.add_argument("--shapes", type=parse_shapes, default=DEFAULT_SHAPES,
                   help="comma-separated MxN list, e.g. 32x768,2048x4096 (default: built-in grid)")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    p.add_argument("--iters", type=int, default=200, help="calls per timed loop / per graph (default 200)")
    p.add_argument("--reps", type=int, default=20, help="timed repetitions; median reported (default 20)")
    p.add_argument("--warmup", type=int, default=50, help="warmup calls before timing (default 50)")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--no-graphs", action="store_true", help="skip the CUDA-graph replay measurement")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: <repo>/benchmarks/out, git-ignored)")
    p.add_argument("--verbose", action="store_true", help="print tracebacks for candidate errors")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out if args.out is not None else repo_root / "benchmarks" / "out"

    if not torch.cuda.is_available():
        print("bench_layernorm.py: torch.cuda.is_available() is False - a CUDA GPU is required.")
        print("Nothing was measured. Run this on a machine with an NVIDIA GPU and a CUDA build of torch.")
        return 1
    try:
        import fused_layernorm_cuda as ext_module  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, or a loader error from a broken .so
        print(f"bench_layernorm.py: cannot import fused_layernorm_cuda ({type(exc).__name__}: {exc}).")
        print(BUILD_HELP)
        return 1
    if not hasattr(ext_module, "layernorm"):
        print("bench_layernorm.py: fused_layernorm_cuda has no `layernorm` function; rebuild the extension.")
        print(BUILD_HELP)
        return 1

    meta = collect_metadata(repo_root, ext_module)
    peak = meta["gpu_peak_bandwidth_gb_s"]
    print(f"GPU: {meta['gpu_name']} ({meta['gpu_sm_count']} SMs); torch {meta['torch_version']}; "
          f"CUDA {meta['cuda_version']}; extension {meta['extension_version']}; commit {meta['git_commit']}")
    if peak is None:
        print("peak unknown; utilization not computed")
    else:
        print(f"peak bandwidth used for utilization: {peak} GB/s (table in this script; check it for your card)")

    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        init_profiler_once()  # must precede the first CUDA-graph capture; see its docstring
        for m, n in args.shapes:
            print(f"- {m}x{n} {args.dtype} ...", flush=True)
            results.append(bench_shape(ext_module, m, n, args.dtype, args, peak))

    table = render_table(results, peak)
    print()
    print(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"bench_{stamp}_{_gpu_slug(meta['gpu_name'])}_{args.dtype}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    payload = {
        "metadata": meta,
        "settings": {
            "iters": args.iters, "reps": args.reps, "warmup": args.warmup, "eps": args.eps,
            "dtype": args.dtype, "graphs": not args.no_graphs,
            "shapes": [list(s) for s in args.shapes],
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(meta, table, args), encoding="utf-8")
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
