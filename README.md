# LayerNorm CUDA kernel for PyTorch (forward-only)

A small, readable CUDA implementation of LayerNorm (plus an optional fused GELU epilogue) exposed to
PyTorch as a C++ extension, together with an honest account of what has and has not been measured.

> **Status of this README (rewritten 2026‑08‑18).** An earlier version of this page opened with
> "Making LayerNorm **1.86-2.66x faster** by REMOVING optimizations" and quoted 10.7 % bandwidth
> utilisation among other figures. A line‑by‑line audit found that those headline numbers match
> **no committed data file**, that the committed measurements record **host launch/dispatch overhead
> rather than kernel time**, and that both usage snippets raised `TypeError` as written. Everything
> below is limited to what the code and the committed data support. The full list of removed or
> corrected claims is in [`CHANGELOG.md`](CHANGELOG.md).
>
> **Update 2026-08-20 (v0.3.0).** The rewritten code has now been built and measured on an
> A100-SXM4-40GB: the full test suite (102 tests) passes, `bench_layernorm.py` has real profiler
> data ([`benchmarks/results/2026-08-20_a100-40gb_kernel_time/`](benchmarks/results/2026-08-20_a100-40gb_kernel_time/)),
> and a vectorised single-pass (Welford) kernel variant was added so the comparison with PyTorch's
> kernel is one of equals. See "What has been measured".

## What this is

* `csrc/layernorm_cuda_kernel.cu` — two templated CUDA kernels, both **one thread block per row**,
  block reduction with warp shuffles, optional affine (`weight`, `bias`, each independently
  optional), optional GELU epilogue (erf form by default, tanh approximation on request):
  * a scalar **two‑pass** kernel (mean, then centred sum of squares) — the simple, readable
    baseline, used for odd/tiny row lengths and for small row counts, where it measures faster;
  * a **vectorised single‑pass** kernel (16‑byte loads — 4 floats / 8 halves — and Welford
    statistics merged with Chan's parallel update), used when `N` is a multiple of the vector
    width, `N ≥ 128` and there are ≥ 256 rows — the same algorithm PyTorch's
    `vectorized_layer_norm_kernel` uses, which is what makes the head‑to‑head comparison fair.

  Accumulation is in `float` for `float16/bfloat16/float32` inputs and in `double` for `float64`
  inputs. Launched on the current PyTorch CUDA stream under a device guard, with a launch‑error
  check. The selection heuristic and its measured basis are commented in the launcher.
* `csrc/bindings.cpp` — argument validation (`TORCH_CHECK`), flattening of any rank ≥ 1 input to
  `(rows, N)`, reshape back, `pybind11` bindings with keyword arguments.
* `fused_layernorm/` — a tiny Python package: `layer_norm()` / `layer_norm_gelu()` with the same
  signature as `torch.nn.functional.layer_norm`, an `nn.LayerNorm` subclass, and a `replace_layernorm()`
  helper. **The kernel is forward‑only.** When autograd is needed (grad mode on and any input requires
  grad) the package falls back to `torch.nn.functional.layer_norm`, so training code keeps working
  and simply does not use the kernel.
* `benchmarks/bench_layernorm.py` — a benchmark that separates *eager per‑call latency*, *GPU kernel
  time* (via `torch.profiler`), and *CUDA‑graph replay time*, and records the environment. See
  [`docs/methodology.md`](docs/methodology.md) for why those three numbers differ.

The name "Fused" in the repository name is historical: relative to PyTorch's own single‑kernel
LayerNorm forward, the only fusion here is the optional GELU epilogue.

## Install

Requirements: a CUDA GPU, a PyTorch build with CUDA support (`torch>=2.0`), and the matching CUDA
toolkit with `nvcc` on `PATH` (the extension is compiled by `torch.utils.cpp_extension`).

```bash
git clone https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator.git
cd Fused-LayerNorm-CUDA-Operator
pip install --no-build-isolation -e .   # builds fused_layernorm_cuda against the torch you have installed
                                        # (with build isolation pip would compile against a fresh torch download)
python -m pytest tests -q   # GPU tests run only when a GPU and the extension are available
```

Built and tested with PyTorch 2.13.0+cu129 / CUDA toolkit 12.9 on an NVIDIA A100‑SXM4‑40GB
(all 102 tests pass); their predecessor (commit `12dee09`) was built with PyTorch 2.7.1+cu128 /
CUDA 12.8 on an A100‑SXM4‑80GB in August 2025. Other GPUs/toolkits should build but are unmeasured.

## Usage

Extension module (positional or keyword arguments; `weight` / `bias` may be `None`):

```python
import torch, fused_layernorm_cuda

x = torch.randn(32, 4096, device="cuda")
w = torch.rand(4096, device="cuda") + 0.5
b = torch.randn(4096, device="cuda")

y = fused_layernorm_cuda.layernorm(x, w, b, eps=1e-5)                          # == F.layer_norm(x, (4096,), w, b, 1e-5)
y = fused_layernorm_cuda.layernorm_gelu(x, w, b, eps=1e-5, approximate="none") # == F.gelu(F.layer_norm(...))
```

Python package (falls back to PyTorch whenever the kernel cannot be used):

```python
import torch, torch.nn as nn
import fused_layernorm

fused_layernorm.is_available()                       # True iff the extension imported and CUDA is available
y = fused_layernorm.layer_norm(x, (4096,), w, b)     # same signature as F.layer_norm

model = nn.Sequential(nn.Linear(768, 768), nn.LayerNorm(768), nn.GELU()).cuda().eval()
n = fused_layernorm.replace_layernorm(model)          # swaps each nn.LayerNorm for fused_layernorm.LayerNorm,
                                                     # sharing the existing parameters; returns the count (1)
with torch.inference_mode():
    out = model(torch.randn(8, 128, 768, device="cuda"))
```

Inputs of any rank ≥ 1 are normalised over the last dimension; non‑contiguous inputs are made
contiguous inside the binding. The package uses the kernel only when it can do so without changing
behaviour: CUDA input, 1‑D `normalized_shape` equal to the last dimension, `weight`/`bias` with the
input's dtype and device, no active CUDA autocast region, and no gradient required. Every other call
goes to `F.layer_norm` and behaves exactly as `nn.LayerNorm` would (under autocast that means an fp32
result; outside autocast PyTorch's CUDA LayerNorm rejects a weight/bias dtype different from the
input, and so does this kernel).

One gotcha worth knowing: in `eval()` mode under `no_grad`/`inference_mode`, PyTorch's
`nn.TransformerEncoderLayer` (and `nn.MultiheadAttention`) take a fused "fast path" that reads
`norm1`/`norm2`'s parameters directly and never calls the sub‑modules' `forward()`, so a replaced
LayerNorm inside such a layer is bypassed unless you disable that path
(`torch.backends.mha.set_fastpath_enabled(False)`). `examples/example_usage.py` shows this.

## What has been measured

### 2026‑08‑20, A100‑SXM4‑40GB: GPU kernel time for the code in this tree

[`benchmarks/results/2026-08-20_a100-40gb_kernel_time/`](benchmarks/results/2026-08-20_a100-40gb_kernel_time/)
(PyTorch 2.13.0+cu129, CUDA 12.9, produced by `benchmarks/bench_layernorm.py`; fp32 and fp16 files).
`kernel_us` is device‑side kernel time from `torch.profiler` over 200 calls — the only number here
that says anything about the kernel itself. fp32:

| rows × N | `F.layer_norm` kernel_us | ours kernel_us | kernel ratio | eager: `nn.LayerNorm` → ours |
|---|---:|---:|---:|---|
| 32 × 768 | 6.00 | 4.53 | **1.32×** | 14.4 → 6.4 µs |
| 32 × 1024 | 5.16 | 3.99 | **1.29×** | 19.5 → 8.8 µs |
| 32 × 4096 | 8.81 | 5.79 | **1.52×** | 19.8 → 8.9 µs |
| 17 × 1023 | 7.20 | 3.66 | **1.97×** | 23.4 → 9.0 µs |
| 128 × 4096 | 10.50 | 7.51 | **1.40×** | 19.8 → 8.8 µs |
| 512 × 1024 | 6.86 | 7.80 | 0.88× | 19.9 → 8.7 µs |
| 2048 × 4096 | 67.81 | 62.71 | **1.08×** | 68.8 → 63.6 µs |
| 4096 × 4096 | 140.09 | 119.75 | **1.17×** | 141.4 → 123.2 µs |
| 8192 × 1024 | 57.65 | 59.04 | 0.98× | 58.5 → 59.9 µs |
| 16384 × 768 | 87.86 | 89.41 | 0.98× | 87.9 → 89.5 µs |
| 4096 × 12288 | 456.24 | 446.15 | **1.02×** | 457.5 → 447.1 µs |

How to read it:

* **Small/latency‑bound shapes (≲ 2 MiB): the kernel is 1.3–2.0× faster than PyTorch's** and the
  eager per‑call latency is 2.2–2.6× lower than `nn.LayerNorm` (the eager gap is mostly dispatch
  and allocation overhead — `nn.LayerNorm` also allocates `mean`/`rstd` — which is the effect the
  2025 numbers below measured without knowing it). The 17 × 1023 row is where PyTorch falls back to
  its two‑kernel path (odd N); this kernel handles odd N in one launch.
* **Large memory‑bound shapes: mostly at or ahead of parity** (1.02–1.17× at N = 4096/12288;
  effective bandwidth 1 071 vs 990 GB/s at 2048 × 4096, 1 121 vs 958 at 4096 × 4096, 903 vs 883 at
  4096 × 12288 — 58–72 % of the 1 555 GB/s datasheet peak for this kernel),
  **slightly behind on narrow rows with many of them** (0.88–0.98× at N = 768/1024).
  The remaining loss cases are block‑scheduling, not algorithmic: both sides run the same
  single‑pass Welford + 16‑byte‑load algorithm there.
* fp16 (same directory): faster or at parity everywhere except 512 × 1024 (0.69×), 2048 × 4096
  (0.89×) and 4096 × 4096 (0.97×); at 4096 × 12288 it is 1.10× ahead.
* CUDA‑graph capture works (`graph_us` in the JSON tracks `kernel_us` closely for every shape),
  which also retires the old "cannot be captured" caveat.
* The August‑2025 claim this repository once made — "1.86–2.66× faster" — was an eager‑latency
  ratio. On this GPU the measured eager ratio against `nn.LayerNorm` is 2.2–2.6× below ~2 MiB, so
  the *number* was reproducible; what was wrong was calling it kernel speed. The kernel‑time
  ratios above are the defensible version of the claim, and they required adding the vectorised
  kernel: the two‑pass scalar kernel alone measured 0.42–0.98× at the large shapes.

### 2025‑08‑17, A100‑SXM4‑80GB: eager latency only (historical, predecessor kernel)

There is one older set of committed measurements, taken with the **predecessor** of
the kernel in this tree (commit `12dee09`: same block‑per‑row, two‑pass design, but fixed
256/512/1024‑thread blocks, fp32 accumulation for every dtype, the legacy default stream, and
positional‑only bindings):
[`benchmarks/results/2025-08-17_a100_eager_latency/`](benchmarks/results/2025-08-17_a100_eager_latency/)
(NVIDIA A100‑SXM4‑80GB, PyTorch 2.7.1+cu128, CUDA 12.8, fp32, produced by the scripts now in
`benchmarks/legacy/`). Both modes time **the whole call from Python** with CUDA events —
`nn.LayerNorm.__call__` versus a raw `pybind11` call to the extension:

| rows × N | input size | `nn.LayerNorm` single call | ours single call | `nn.LayerNorm` pipelined | ours pipelined |
|---|---:|---:|---:|---:|---:|
| 1 × 768 | 3 KiB | 119.8 µs | 37.7 µs | 21.11 µs | 10.06 µs |
| 8 × 768 | 24 KiB | 50.2 µs | 29.8 µs | 21.12 µs | 10.14 µs |
| 32 × 768 | 96 KiB | 50.9 µs | 31.5 µs | 21.17 µs | 10.11 µs |
| 32 × 1024 | 128 KiB | 48.5 µs | 30.1 µs | 21.17 µs | 10.03 µs |
| 32 × 2048 | 256 KiB | 47.0 µs | 29.5 µs | 22.08 µs | 10.10 µs |
| 32 × 4096 | 512 KiB | 47.0 µs | 28.2 µs | 21.09 µs | 9.99 µs |
| 64 × 4096 | 1 MiB | 55.1 µs | 28.5 µs | 21.06 µs | 10.11 µs |
| 128 × 4096 | 2 MiB | 49.1 µs | 28.9 µs | 21.36 µs | 10.10 µs |
| 17 × 1023 | 68 KiB | 58.2 µs | 40.1 µs | 24.21 µs | 10.01 µs |
| 32 × 12288 | 1.5 MiB | 54.5 µs | 30.0 µs | 24.38 µs | 12.46 µs |

*"single call"*: one call between two CUDA events on an idle GPU, 50 samples, mean; *"pipelined"*:
500 back‑to‑back calls between two events, per‑call mean; both under `torch.set_grad_enabled(False)`.
Numbers are copied from `publication_validation_results.json` (ms × 1000). A second file from the
same day, `publication_results.json`, was produced **with autograd enabled** for the PyTorch side and
gives 26.3–29.7 µs vs 10.6–10.8 µs pipelined for five of these shapes.

### How to read that table

* The per‑call time is **flat** — ≈10 µs for the extension and ≈21 µs for `nn.LayerNorm` — from a
  3 KiB input to a 2 MiB one (≈680× more data). In a pipelined loop the per‑call time is roughly
  max(host time to issue one call, GPU time to run one call); a time that does not move with the
  data size is the signature of a **host‑bound** measurement (Python → `nn.Module.__call__` →
  dispatcher → allocation → kernel launch). On that reading the measured ratio (≈2.1× pipelined,
  ≈1.8× single‑call in this file) is a difference in per‑call *overhead* — a raw `pybind11` call that
  allocates one output vs. the full `nn.LayerNorm` path that also allocates `mean`/`rstd` (and builds
  an autograd node when grad is enabled) — and **not evidence about kernel speed**. The 2026
  profiler data above confirms the reading: at these sizes the kernels themselves take 4–9 µs, so
  the flat ≈10 µs / ≈21 µs pipelined numbers were indeed dominated by per‑call overhead.
* Within this 2025 data, kernel execution time was **not** isolated by any measurement (no
  profiler, `nsys`, `ncu`, or CUDA‑graph run exists for the predecessor kernel); only the 2026
  section above supports statements about kernel speed.
* The only "bandwidth" figure in the committed data is 26.4 GB/s from a single event‑timed 32 × 4096
  call with no warm‑up loop of its own (`bandwidth` block of the same JSON). Against the A100‑SXM4‑80GB peak of **2 039 GB/s**
  (NVIDIA datasheet; the previously quoted 1 555 GB/s is the 40 GB part) that is 1.3 %; the pipelined
  9.99 µs would imply ~108 GB/s (5.3 %). Both are dominated by launch overhead and say little about the
  kernel.
* At production shapes (thousands of rows per call) LayerNorm is memory‑bound. The expectation
  recorded here in the 0.2.0 rewrite — that a two‑pass kernel with scalar loads would not beat
  PyTorch in that regime — was **confirmed** by the first hardware run (0.42–0.98× at the large
  shapes), and is why 0.3.0 added the vectorised single‑pass kernel, which is the one measured in
  the 2026 section above.
* Numerical agreement with PyTorch: max |ours − `nn.LayerNorm`| = 4.77 × 10⁻⁷ at 32 × 4096 fp32
  (`accuracy` block). This is a discrepancy versus PyTorch, not "better accuracy"; it was measured with
  the default `weight = 1, bias = 0`. The rewritten test suite uses random affine parameters.

### What PyTorch's kernel does (for a fair comparison)

PyTorch v2.7.1, `aten/src/ATen/native/cuda/layer_norm_kernel.cu` (1 502 lines; the forward path is
roughly 460 of them and also handles fp16/bf16/fp32/fp64, optional affine, `mean`/`rstd` outputs for
autograd, and ROCm): for fp32/fp16/bf16 with `N % 4 == 0`, `N ≤ 2²⁴` and pointers aligned to
4·sizeof(dtype) bytes (16 B fp32, 8 B fp16/bf16) it launches one 128‑thread block per row using
4‑wide vectorised loads and single‑pass Welford statistics; otherwise it falls back to two launches
(`RowwiseMomentsCUDAKernel` then `LayerNormForwardCUDAKernel`). The 2026 profiler run records
exactly those two kernels at 17 × 1023 (7.2 µs total vs 3.7 µs for this extension's single
launch — the biggest win in the table) and the single `vectorized_layer_norm_kernel` everywhere
else.

## Not verified / limitations

* Any GPU other than one A100‑SXM4‑40GB (2026 data) and one A100‑SXM4‑80GB (2025 eager data);
  any PyTorch/CUDA other than 2.13.0 / 12.9 (2026) and 2.7.1 / 12.8 (2025). The block‑size and
  kernel‑selection heuristics in the launcher are tuned on the A100 measurements only.
* On the measured GPU the kernel is *slower* than PyTorch's at some many‑narrow‑row shapes
  (fp32: 0.88× at 512 × 1024, 0.98× at 8192 × 1024 and 16384 × 768; fp16 additionally 0.89× at
  2048 × 4096). Faster‑everywhere is not claimed.
* No backward pass; the package falls back to PyTorch when gradients are required.
* GELU epilogue: numerics are tested against `F.gelu` on hardware (both forms), but its *speed*
  has not been benchmarked. Note the previous version used the tanh approximation while calling it
  "GELU"; the two differ by up to ~4.7 × 10⁻⁴.
* bf16 and fp64 pass the test suite on hardware but have no committed benchmark (fp32/fp16 do).

## Reproduce / benchmark

```bash
python benchmarks/bench_layernorm.py                    # writes benchmarks/out/*.json and prints a table
python benchmarks/bench_layernorm.py --dtype float16 --shapes 32x768,4096x4096
```

The script reports, per shape: eager per‑call latency (`nn.LayerNorm`, `F.layer_norm`, extension),
GPU kernel time from `torch.profiler` with the kernel names seen, CUDA‑graph replay time, and effective
bandwidth against a per‑device peak table. Only the profiler/graph numbers can support a claim about
the kernel. To publish results, copy the JSON into `benchmarks/results/<date>_<gpu>_<what>/` with a
README stating the command, commit hash and environment (see `benchmarks/README.md`).

## Tests

`tests/test_layernorm.py` (pytest) compares the extension against `torch.nn.functional.layer_norm`
with **random** `weight`/`bias` across shapes from 1 × 1 to 2048 × 4096 (odd, prime, tiny, large N),
dtypes fp32/fp64/fp16/bf16 (fp64 includes an ill‑conditioned‑row case), weight‑only/bias‑only/neither,
rank‑1/3/4 inputs, non‑contiguous inputs, error cases, both GELU forms, execution on a side stream,
determinism, forward‑only semantics, and empty batches. GPU tests are skipped automatically when the
extension or a GPU is missing; the CPU‑only tests (fallback path, `replace_layernorm`) always run.
All 102 tests pass on an A100‑SXM4‑40GB with PyTorch 2.13.0+cu129 (2026‑08‑20; both kernel paths
are exercised — the vectorised kernel needs ≥ 256 rows and the suite includes 2048 × 4096).

## Project layout

```
csrc/                      layernorm_cuda_kernel.cu, bindings.cpp, layernorm.h
fused_layernorm/           Python package (layer_norm, layer_norm_gelu, LayerNorm, replace_layernorm)
tests/                     pytest suite
examples/example_usage.py
benchmarks/bench_layernorm.py           the benchmark (eager / profiler kernel time / CUDA graph)
benchmarks/legacy/                       the August‑2025 scripts, unmodified, with a README of their flaws
benchmarks/results/2026-08-20_a100-40gb_kernel_time/  profiler + graph data for the current kernels (fp32, fp16)
benchmarks/results/2025-08-17_a100_eager_latency/     eager‑latency data (predecessor kernel, commit 12dee09)
benchmarks/results/historical_2025-07_deleted_kernel/ July‑2025 files about a kernel that no longer exists
docs/methodology.md        how to time a small CUDA op from PyTorch without fooling yourself
CHANGELOG.md               what was claimed before, what was removed, and why
```

## Contributing

The most useful contributions are measurements: run `benchmarks/bench_layernorm.py` and the test suite
on any non‑A100 GPU and open a PR with the JSON and environment details. Also welcome: a backward
kernel, tuning for the shapes where PyTorch still wins (many narrow rows), and CI that runs on a GPU.

## Citation

```bibtex
@software{shrivastava2025layernorm_cuda,
  title  = {A simple LayerNorm CUDA kernel for PyTorch},
  author = {Chinmay Shrivastava},
  year   = {2025},
  url    = {https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
