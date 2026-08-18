# Timing a small CUDA op from PyTorch — methodology

This note explains how the numbers in `benchmarks/` are (and were) produced,
which of them mean what, and what PyTorch's own LayerNorm kernel does, so that
a reader can judge any speedup claim made about this repository. It contains
no measurements of its own; every number quoted is either an external
specification (cited) or copied from a committed file (cited by path).

## 1. What a CUDA-event pair around one Python call measures

`torch.cuda.Event.record()` enqueues a timestamp marker on the current stream;
`start.elapsed_time(end)` (CUDA `cudaEventElapsedTime`, resolution around
0.5 µs per the CUDA Runtime API docs) is the difference between the two
GPU-side timestamps.

```python
torch.cuda.synchronize()          # GPU idle
start.record()                    # marker executes immediately
y = f(x)                          # Python -> dispatcher -> launch -> kernel
end.record()
torch.cuda.synchronize()
start.elapsed_time(end)
```

Because the GPU is idle when `start` is recorded, the marker executes at
once, and the GPU then *waits* for the host to get around to launching the
kernel. The `end` marker is likewise enqueued by the host and executes as soon
as the GPU reaches it, so the measured interval is roughly

    max( host time from start.record() until end.record() reaches the GPU ,
         host time to launch the kernel + kernel time )

i.e. it includes Python overhead, dispatcher overhead and CUDA launch latency —
typically tens of microseconds in total — and only exposes the kernel's own
duration when the kernel is longer than that host work. For a kernel that
itself takes a few microseconds, the host term dominates what is measured.
Anything that changes host-side cost (autograd on/off, whether the events
already exist, whether the call is the first in the process, whether the other
implementation ran just before) changes the number, and the ratio between two
implementations timed this way is a ratio of host overheads.

This is exactly how the "realistic" numbers in
[`../benchmarks/results/2025-08-17_a100_eager_latency/`](../benchmarks/results/2025-08-17_a100_eager_latency/README.md)
were taken (see [`../benchmarks/legacy/README.md`](../benchmarks/legacy/README.md)).

## 2. What a loop of launches measures, and the host-bound signature

```python
start.record()
for _ in range(iters):
    y = f(x)
end.record(); torch.cuda.synchronize()
per_call = start.elapsed_time(end) / iters
```

Launches are asynchronous, so the host issues call *k+1* while the GPU is
still running call *k*. In steady state the per-call time is approximately

    max( host time to issue one call , GPU time to run one call )

If the host cannot issue faster than the GPU runs, you are measuring the host.
The signature is easy to recognise: **the per-call time does not change when
the tensor gets bigger.** In `publication_validation_results.json`
(`performance.optimal`) the extension's per-call time is 10.06 µs for 1x768
fp32 (3 KB) and 10.10 µs for 128x4096 (2 MB) — a ~680x increase in data with
no change in time. That number is the host's per-launch cost of a bare pybind
call into a kernel; PyTorch's 21 µs in the same table is the host's cost of
`nn.LayerNorm.forward` through the dispatcher. Neither says how long either
kernel runs. A `time.perf_counter()` loop with a final `synchronize()`
(the "eager" measurement below) has the same property; it is just easier to
reason about because it explicitly measures wall time on the host.

## 3. The three measurements `bench_layernorm.py` reports

| name | method | answers |
|---|---|---|
| **eager per-call latency** | `torch.inference_mode()`, warm-up, `perf_counter()` around `iters` calls + `synchronize()`, median over `reps` | "What does one Python call cost me?" Includes Python/dispatch/launch. Flat with size ⇒ host-bound. Not a kernel number. |
| **GPU kernel time** | `torch.profiler.profile(activities=[ProfilerActivity.CUDA])` over `iters` calls; sum of device-side kernel durations / `iters`; kernel names recorded | "How long does the GPU spend in the kernel(s)?" **The only number that may be quoted as a kernel speedup.** Also shows *which* kernels ran (one vs. two launches). |
| **CUDA-graph replay** | capture `iters` calls into one `torch.cuda.CUDAGraph`, replay timed with CUDA events | "What does the op cost when Python and the dispatcher are out of the way?" Also documents whether the op is capturable at all. |

Effective bandwidth = bytes moved / kernel time, with bytes = M·N·elem·2 +
N·elem·2 (read input, write output, read weight and bias). Utilisation is that
divided by the device's peak (section 6). Bandwidth computed from an eager or
single-call time is not a kernel bandwidth.

## 4. Comparing against the right baseline

* Compare against `torch.nn.functional.layer_norm(x, (N,), w, b, eps)` under
  `torch.inference_mode()`, in the same dtype, with the same (non-identity)
  `w`, `b`. That is the fair functional equivalent of a forward-only kernel.
* Do **not** compare a raw extension call against an `nn.LayerNorm` module
  call with autograd enabled: the module call additionally records an autograd
  node and keeps `mean`/`rstd` for backward. Two of the three legacy scripts
  did this; the one that disabled grad reports a smaller ratio (2.13x vs 2.57x
  in optimal mode, from the two JSON files in
  `benchmarks/results/2025-08-17_a100_eager_latency/`) — though those two runs
  also differ in iteration and warm-up counts, so the gap is not attributable
  to autograd alone.
* Warm up both implementations, alternate or randomise the order, and never
  time the very first call in a process (module loading, context creation,
  cuBLAS/cuDNN handles, allocator growth all land there).
* Report the median of several repetitions, and report the environment (GPU,
  clocks if you can lock them, torch/CUDA versions, CPU — the host CPU
  determines the launch cost).
* Check correctness with random weight and bias before timing. A kernel that
  ignores `weight`/`bias` passes an identity-affine check.

## 5. What PyTorch's own LayerNorm kernel does (v2.7.1)

Source: [`aten/src/ATen/native/cuda/layer_norm_kernel.cu` at tag v2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
(1,502 lines; the forward path — kernels at lines 53–311, launch and dispatch
at 732–860, host wrapper `layer_norm_cuda` at 1371–1422 — is roughly
460 of them; the rest is backward). Facts, with line numbers from that file:

* `kCUDANumThreads = 256` and `vec_size = 4` (lines 35–37).
* `LayerNormKernelImplInternal` (lines 793–838) chooses one of two paths.
  **Fast path** (condition at lines 815–826): dtype is float / half / bfloat16,
  N ≤ 2^24, `N % 4 == 0`, and the input, output, weight and bias pointers are
  aligned to `4 * sizeof(T)` bytes (16 B for fp32, 8 B for fp16/bf16). Then
  `launch_vectorized_layer_norm_kernel` (lines 732–791) launches
  `vectorized_layer_norm_kernel` with **one block per row of 128 threads**
  (`dim3 threads(warp_size, num_threads() / warp_size)`, line 747, with
  `num_threads() = C10_WARP_SIZE * 4` from
  [`thread_constants.h`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/thread_constants.h)),
  4-wide vector loads (`aligned_vector<T, 4>`, i.e. `float4` for fp32) and
  **single-pass Welford** statistics combined with warp shuffles
  (`compute_stats`, lines 156–216). It writes `mean` and `rstd` for every row
  (lines 279–280) because autograd needs them; the host wrapper allocates
  those two tensors in the accumulate type (lines 1400–1401).
* **Fallback** (lines 828–837), used for float64 (excluded by the fast-path
  dtype test; the double overload of the vectorised impl at lines 284–296 is a
  compile-only stub), for `N % 4 != 0`, and for misaligned pointers:
  **two launches**, `RowwiseMomentsCUDAKernel` (lines 53–90) with
  `cuda_utils::kCUDABlockReduceNumThreads` = 512 threads per row
  ([`block_reduce.cuh`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/block_reduce.cuh)
  line 10) computing mean/rstd with a Welford block reduce, then
  `LayerNormForwardCUDAKernel` (lines 92–112) with `kCUDANumThreads` = 256
  threads per row applying them.
* Accumulation is in `acc_type<T, true>` (float for fp32/fp16/bf16, double for
  fp64), selected in `LayerNormKernelImpl` (lines 840–860).

So when you profile `F.layer_norm` you should expect to see
`vectorized_layer_norm_kernel` for aligned fp32/fp16/bf16 rows with N % 4 == 0
(e.g. 32x768, 32x4096) and `RowwiseMomentsCUDAKernel` +
`LayerNormForwardCUDAKernel` otherwise (e.g. 17x1023, or any float64 input).
`bench_layernorm.py` records the kernel names for exactly this reason.

## 6. What to expect from a block-per-row kernel, and peak bandwidths

LayerNorm does only a handful of flops per element read/written (O(N) work
for a row of N elements), so at large row counts it is memory-bound. The only large-row timings in the repository are
for the *July 2025* kernel and PyTorch
([`../benchmarks/results/historical_2025-07_deleted_kernel/large_model_results.csv`](../benchmarks/results/historical_2025-07_deleted_kernel/large_model_results.csv), see that folder's README for provenance caveats):
PyTorch took 0.056–0.375 ms for 4096 rows x 1600–12288 columns, which — if
those runs were fp32 (the dtype is not recorded) — is roughly 1 TB/s of
read+write traffic on an A100-SXM4-80GB, i.e. about half of the 2039 GB/s
peak. In that regime a simple block-per-row kernel that reads each row three
times (mean pass, variance pass, normalise pass; PyTorch's vectorised kernel
reads it twice) with scalar loads is — as an unmeasured expectation, not a
result — unlikely to beat PyTorch and may be slower. This is **unmeasured** for
the current kernel: `bench_layernorm.py` includes 2048x4096, 4096x4096,
8192x1024, 16384x768 and 4096x12288 in its default grid precisely so that this
regime gets measured and committed.

At small sizes (tens of rows) both kernels are expected to be short relative
to the ~10-21 µs per-call host cost seen in
`benchmarks/results/2025-08-17_a100_eager_latency/` (kernel time itself was
not measured there), so everything is dominated by launch and dispatch cost,
where a bare pybind call is cheaper than the PyTorch dispatcher; the eager
numbers in that folder show that and nothing else.

Peak HBM bandwidth (NVIDIA A100 datasheet, "GPU Memory Bandwidth"):

| device | GB/s |
|---|---|
| A100 40 GB (PCIe and SXM) | 1,555 |
| A100 80 GB PCIe | 1,935 |
| A100 80 GB SXM | 2,039 |

`bench_layernorm.py` looks the peak up by `torch.cuda.get_device_name()`
substring (plus H100 SXM 3,350 / H100 PCIe 2,000 / V100 900 from the
respective NVIDIA datasheets) and prints "peak unknown; utilization not
computed" for anything else. Do not hard-code 1555 for "any A100".

## Sources

* NVIDIA A100 Tensor Core GPU datasheet — <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf> (product page: <https://www.nvidia.com/en-us/data-center/a100/>)
* CUDA Runtime API, event management (`cudaEventRecord`, `cudaEventElapsedTime`) — <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html>
* PyTorch benchmark recipe (`torch.utils.benchmark`, warm-up, synchronisation) — <https://pytorch.org/tutorials/recipes/recipes/benchmark.html>
* PyTorch profiler — <https://pytorch.org/docs/stable/profiler.html>
* PyTorch CUDA graphs — <https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
* PyTorch v2.7.1 `layer_norm_kernel.cu` — <https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu>
