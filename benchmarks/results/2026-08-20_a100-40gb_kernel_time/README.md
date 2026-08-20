# 2026-08-20 — A100-SXM4-40GB — kernel-time measurements for v0.3.0

First hardware measurements of the kernels in this tree: the scalar two-pass
kernel plus the vectorised single-pass Welford kernel. Unlike the 2025 data,
these runs report **GPU kernel time from `torch.profiler`** and CUDA-graph
replay time, not just eager latency.

Three files, all produced from a **clean clone of commit `73740ed`** (each
JSON's `metadata` records `git_commit = 73740ed…`, `git_dirty = false`,
`extension_version = 0.3.0`, NVIDIA driver 570.148.08 and
`TORCH_CUDA_ARCH_LIST = 8.0`, plus torch/CUDA/CPU/Python details):

| file | command | what it is |
|---|---|---|
| `bench_…_211216_…_float32.*` | `python benchmarks/bench_layernorm.py` | fp32, automatic kernel selection — the numbers the top-level README quotes |
| `bench_…_211234_…_float16.*` | `python benchmarks/bench_layernorm.py --dtype float16` | fp16, automatic kernel selection |
| `bench_…_211303_…_float32.*` | `FUSED_LAYERNORM_FORCE_KERNEL=scalar python benchmarks/bench_layernorm.py` | fp32 **scalar-kernel baseline**: the two-pass scalar kernel forced on every shape (the JSON's kernel names show `layernorm_kernel` throughout), backing the claim about why the vectorised kernel exists |

Environment: NVIDIA A100-SXM4-40GB (108 SMs, 1 555 GB/s datasheet peak),
PyTorch 2.13.0+cu129, CUDA toolkit 12.9, AMD EPYC 7J13 host, Linux
6.8.0-60-generic. Idle GPU, default clocks; the full pytest suite (104 tests) passed on the same clean-clone
build. Method per shape: candidates
warmed up then timed interleaved; `eager_us` = median of 20 reps of 200 calls,
`kernel_us` = profiler device-kernel time / 200 calls, `graph_us` = CUDA-graph
replay of 200 captured calls (CUDA events).

Summary (`kernel_us`, ours vs `F.layer_norm`):

* fp32, automatic selection: faster on 9 of 11 shapes (1.01–1.97×), parity at
  8192×1024 (1.00×), slower only at 512×1024 (0.93×).
* fp16: faster on 10 of 11 shapes (1.02–1.98×), slower only at 512×1024
  (0.74×).
* Scalar baseline at the M ≥ 512 shapes: 0.42–0.98× — the measured reason the
  vectorised kernel was added. (At the M ≤ 128 shapes the scalar kernel wins,
  1.29–1.97×, which is why the launcher still selects it there.)
* Eager per-call latency vs `nn.LayerNorm` below ~2 MiB: 2.2–2.6× lower in
  fp32, 2.1–2.5× in fp16 (host-overhead bound; see `docs/methodology.md`).
* `graph_us` stays within 13 % of `kernel_us` for every shape (usually below
  it), which also documents that CUDA-graph capture works.

Every number quoted in the top-level README's 2026 section is taken from
these JSONs. An earlier same-day pair of files (20:12, removed) was produced
from a dirty pre-release working tree; this directory replaces them so that
committed data and committed code correspond exactly.
