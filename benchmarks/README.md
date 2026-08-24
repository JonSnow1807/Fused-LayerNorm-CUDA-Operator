# Benchmarks

This directory contains two benchmark scripts — `bench_layernorm.py` (the
plain-LayerNorm benchmark whose committed v0.3.0 data it produced) and
`bench_norms.py` (the v0.4.0 op family: RMSNorm, fused residual-add, fp8,
training step; built on the same timing core) — the legacy scripts that
produced the numbers in the old README (`legacy/`, kept byte-identical for
provenance), and committed result files (`results/`).

## Status of the numbers in this repository (read this first)

* **Kernel-time, CUDA-graph and bandwidth measurements exist since 2026‑08‑20:**
  [`results/2026-08-20_a100-40gb_kernel_time/`](results/2026-08-20_a100-40gb_kernel_time/README.md)
  holds `bench_layernorm.py` output (fp32 and fp16, plus a scalar-kernel
  baseline run) for the v0.3.0 kernels on an A100‑SXM4‑40GB, and
  [`results/2026-08-24_a100-40gb_v040_ops/`](results/2026-08-24_a100-40gb_v040_ops/README.md)
  holds `bench_norms.py` output for the v0.4.0 op family; each README states
  the commands, commit and environment. Only these profiler-based files speak
  about kernel speed.
* The eager per-call latencies in
  [`results/2025-08-17_a100_eager_latency/`](results/2025-08-17_a100_eager_latency/README.md)
  were taken with the predecessor of the current kernel (commit `12dee09`; same
  block-per-row two-pass design, different block sizes / accumulation / stream).
  They were produced by the legacy scripts and measure host dispatch + launch
  cost, not kernel time (the single 26.4 GB/s figure there is derived from a
  host-bound single-call time); the README in that folder explains what they do
  and do not show.
* [`results/historical_2025-07_deleted_kernel/`](results/historical_2025-07_deleted_kernel/README.md)
  describes a different, since-deleted kernel and applies to nothing in the
  current tree.

## Running the benchmark

Requirements: a CUDA GPU, a CUDA build of `torch`, and the extension built
(`pip install --no-build-isolation -e .` in the repository root). Nothing else — the script uses only
`torch` and the Python standard library.

```bash
python benchmarks/bench_norms.py                           # v0.4.0 op family, fp16, all ops
python benchmarks/bench_norms.py --op fused_add_rms_norm --dtype float32
python benchmarks/bench_layernorm.py                       # plain LayerNorm, default grid, float32
python benchmarks/bench_layernorm.py --dtype float16
python benchmarks/bench_layernorm.py --shapes 32x768,4096x4096 --iters 500 --reps 30
python benchmarks/bench_layernorm.py --no-graphs           # skip CUDA-graph capture
python benchmarks/bench_layernorm.py --out /tmp/ln-bench   # default: benchmarks/out/ (git-ignored)
python benchmarks/bench_layernorm.py --warmup 100 --eps 1e-6 --verbose   # warm-up calls, eps, tracebacks
```

Flags: `--shapes MxN,MxN,...`, `--dtype {float32,float16,bfloat16,float64}`, `--iters` (calls per timed
loop / per graph, default 200), `--reps` (timed repetitions, median reported, default 20), `--warmup`
(default 50), `--eps` (default 1e-5), `--no-graphs`, `--out DIR`, `--verbose`.

Without a GPU or without the extension the script prints what is missing and
exits with status 1; it never prints a table it did not measure.

The default shape grid is (rows x N): 32x768, 32x1024, 32x4096, 17x1023,
128x4096, 512x1024, 2048x4096, 4096x4096, 8192x1024, 16384x768, 4096x12288 —
i.e. it includes the large-row regime that the legacy scripts never measured
for any version of this kernel.

Before timing each shape the script checks `fused_layernorm_cuda.layernorm`
against `torch.nn.functional.layer_norm` with random, non-identity weight and
bias, and skips the shape (recording why) if the check fails.

## The three measurements — and which one you may quote

For every shape and for each of three candidates (`nn.LayerNorm` module call,
`F.layer_norm`, `fused_layernorm_cuda.layernorm`; all under
`torch.inference_mode()`) the script reports:

| column | what it is | what it tells you |
|---|---|---|
| `eager us/call` | `time.perf_counter()` around a loop of `--iters` calls + `torch.cuda.synchronize()`, median over `--reps` (repetitions interleaved round-robin across candidates after warming all of them) | Python + dispatcher + launch overhead + kernel time. At small sizes the overhead dominates and the number is nearly flat with size. This is what a user pays per Python call. It is what the legacy scripts measured. **It says almost nothing about the kernel.** |
| `kernel us/call` | sum of device-side kernel durations from `torch.profiler` (CUDA activity) over `--iters` calls, divided by `--iters`; kernel names are listed | Time the GPU actually spends in the kernel(s). **This is the only number that may be quoted as a "kernel speedup".** |
| `graph us/call` | `--iters` calls captured into one `torch.cuda.CUDAGraph`, replay timed with CUDA events | Cost of the op when launched from a graph (no Python/dispatch). Also documents whether the candidate is graph-capturable; capture failures are recorded as an error string instead of a number. |

Effective bandwidth (`GB/s`) and `% peak` are computed from `kernel us/call`
only, with bytes = M·N·elem·2 + N·elem·2 (read x, write y, read weight and
bias). The peak comes from a small table keyed on
`torch.cuda.get_device_name()` (A100-SXM4-80GB 2039, A100-SXM4-40GB /
A100-PCIE-40GB 1555, A100 80GB PCIe 1935, H100 80GB HBM3 3350, H100 PCIe 2000,
V100 900 GB/s — NVIDIA datasheets); for any other device the script prints
"peak unknown; utilization not computed" rather than guessing.

See [`../docs/methodology.md`](../docs/methodology.md) for why these
distinctions matter and how to recognise a host-bound measurement.

## Committing results

Results are only meaningful together with the environment they were measured
in. To commit a run:

1. Run the script; it writes `benchmarks/out/bench_<timestamp>_<gpu-slug>_<dtype>.json`
   and a matching `.md` (the `out/` directory is git-ignored on purpose).
2. Create `benchmarks/results/<YYYY-MM-DD>_<gpu-slug>_<what>/`, e.g.
   `benchmarks/results/2025-09-01_a100-sxm4-80gb_kernel-time/`.
3. Copy the JSON (and, if you like, the `.md` table) into that folder.
4. Add a `README.md` in that folder that states, at minimum:
   * the exact command line used,
   * the git commit hash of the tree that was benchmarked (the JSON's
     `metadata.git_commit`; `metadata.git_dirty` records whether the working
     tree had uncommitted changes — it should be `false`),
   * the environment: GPU name and SM count, `torch` version, CUDA version,
     driver version if known, CPU model, Python version (all but the driver are
     in `metadata` in the JSON),
   * a table copied from the JSON (do not retype numbers from memory), and
   * one or two sentences on what the numbers do and do not show
     (host-bound vs. kernel-bound; see the methodology doc).
5. Do not edit committed JSON files afterwards; add a new folder for a new run.

Numbers quoted in the top-level README must point at a file under
`benchmarks/results/` that contains them.
