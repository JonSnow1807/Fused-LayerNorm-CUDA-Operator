# 2026-08-20 — A100-SXM4-40GB — first measurements of the rewritten kernel

First hardware run of the kernels in this tree (v0.3.0: the scalar two-pass
kernel plus the vectorised single-pass Welford kernel added the same day).
Unlike the 2025 data, these runs report **GPU kernel time from
`torch.profiler`** and CUDA-graph replay time, not just eager latency.

* Commands:
  * `python benchmarks/bench_layernorm.py` (fp32 file)
  * `python benchmarks/bench_layernorm.py --dtype float16` (fp16 file)
* Code: the v0.3.0 release commit (the one that added this directory).
* Environment (also recorded in each JSON's `metadata`): NVIDIA A100-SXM4-40GB
  (108 SMs, 1 555 GB/s datasheet peak), driver 570.148.08, PyTorch
  2.13.0+cu129, CUDA toolkit 12.9, `TORCH_CUDA_ARCH_LIST=8.0`, AMD EPYC 7J13
  host, Linux 6.8.0-60-generic. Idle GPU, default clocks; the full pytest
  suite (102 tests) passed on this build before benchmarking.
* Method: per shape, all candidates warmed up then timed interleaved;
  `eager_us` = median of 20 reps of 200 calls (perf_counter around the loop),
  `kernel_us` = sum of device kernel durations from `torch.profiler` / 200
  calls, `graph_us` = CUDA-graph replay of 200 captured calls timed with CUDA
  events. Correctness vs `F.layer_norm` with random affine checked per shape
  before timing.

Summary (kernel_us, ours vs `F.layer_norm`, fp32): faster on 8 of 11 shapes —
1.3-2.0x at latency-bound shapes (32x768 ... 128x4096, 17x1023 where PyTorch
takes its two-kernel fallback), 1.02-1.17x at 2048x4096 / 4096x4096 /
4096x12288 — and slower on 3: 512x1024 (0.88x), 8192x1024 and 16384x768
(0.98x). fp16: faster or at parity everywhere except 512x1024 (0.69x),
2048x4096 (0.89x) and 4096x4096 (0.97x). Eager per-call latency is
1.7-2.6x lower than `nn.LayerNorm` at every shape below ~2 MiB (host-overhead
bound; see docs/methodology.md).
