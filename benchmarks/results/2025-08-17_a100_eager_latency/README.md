# 2025-08-17 — A100 eager per-call latency (legacy scripts)

These two JSON files are the **only committed measurements in the repository**.
They were taken with the predecessor of the kernel now in the tree (the
`3686b4c`/`12dee09` kernel: same block-per-row, two-pass mean/variance,
strided-loop design, but fixed 256/512/1024-thread blocks, fp32 accumulation
for every dtype and the legacy default stream; the current kernel has not been
run anywhere yet). They were produced by the legacy scripts in
[`../../legacy/`](../../legacy/README.md) and they measure **eager per-call
latency** — Python + dispatcher + launch overhead — not kernel execution time.
Read the "How to read these numbers" section before quoting anything.

Every number below is copied from the JSON files in this folder (µs = the
file's ms value x 1000, rounded to 2 decimals). Nothing here was re-measured.

## Provenance

| file | produced by | commit of the tree that ran | recorded environment |
|---|---|---|---|
| `publication_results.json` | [`legacy/publication_ready_benchmark.py`](../../legacy/publication_ready_benchmark.py) | added in `4ea14d2` (2025-08-17); the kernel source is the one introduced in `3686b4c` the same day | the file has no metadata block; the same session's `publication_validation_results.json` recorded the environment below |
| `publication_validation_results.json` | [`legacy/publication_validation.py`](../../legacy/publication_validation.py) | added in `12dee09` (2025-08-17) | `timestamp` 2025-08-17T20:00:04, torch 2.7.1+cu128, CUDA 12.8, NVIDIA A100-SXM4-80GB |

Both scripts call the old positional binding
`fused_layernorm_cuda.layernorm(x, weight, bias, 1e-5)`; the kernel of
`3686b4c` accumulated in `float` for every dtype and launched on the legacy
default stream (see `git show 3686b4c:csrc/layernorm_cuda_kernel.cu`).

## `publication_results.json`

Five configs, fp32. "realistic" = one CUDA-event pair around one Python call,
fresh input each sample, 100 samples, mean ± std, `nn.LayerNorm` module called
**with autograd enabled**. "cached" = one event pair around 1000 back-to-back
launches after 200 warm-ups, divided by 1000.

| config | shape (rows x N) | realistic PyTorch µs (mean ± std) | realistic ours µs (mean ± std) | ratio | cached PyTorch µs | cached ours µs | ratio |
|---|---|---|---|---|---|---|---|
| BERT | 32x768 | 100.95 ± 400.00 | 40.44 ± 40.43 | 2.50 | 26.69 | 10.62 | 2.51 |
| GPT-2 | 32x1024 | 62.13 ± 16.93 | 39.10 ± 10.41 | 1.59 | 26.99 | 10.58 | 2.55 |
| GPT-3 | 32x4096 | 60.33 ± 10.78 | 35.97 ± 5.38 | 1.68 | 26.31 | 10.58 | 2.49 |
| Large Batch | 64x4096 | 64.12 ± 47.36 | 37.17 ± 6.89 | 1.73 | 26.95 | 10.71 | 2.52 |
| Odd Dimensions | 17x1023 | 64.75 ± 17.38 | 36.29 ± 6.23 | 1.78 | 29.65 | 10.75 | 2.76 |

`summary`: `realistic_speedup` 1.8544 (mean of the five ratios),
`cached_speedup` 2.5653. (The config names/shapes are not stored in the file;
they are the `configs` list at lines 86–92 of the script, in order.)

## `publication_validation_results.json`

Ten configs, fp32, `torch.set_grad_enabled(False)`. "realistic" = single-call
event pair, 50 samples, mean; "optimal" = event pair around 500 launches after
100 warm-ups, divided by 500.

| config | shape (rows x N) | realistic PyTorch µs | realistic ours µs | ratio | optimal PyTorch µs | optimal ours µs | ratio |
|---|---|---|---|---|---|---|---|
| Tiny Batch | 1x768 | 119.83 | 37.68 | 3.18 | 21.11 | 10.06 | 2.10 |
| Small Batch | 8x768 | 50.18 | 29.82 | 1.68 | 21.12 | 10.14 | 2.08 |
| BERT | 32x768 | 50.87 | 31.46 | 1.62 | 21.17 | 10.11 | 2.09 |
| GPT-2 Small | 32x1024 | 48.54 | 30.11 | 1.61 | 21.17 | 10.03 | 2.11 |
| GPT-2 Medium | 32x2048 | 47.04 | 29.51 | 1.59 | 22.08 | 10.10 | 2.19 |
| GPT-3 | 32x4096 | 46.96 | 28.18 | 1.67 | 21.09 | 9.99 | 2.11 |
| Large Batch | 64x4096 | 55.13 | 28.55 | 1.93 | 21.06 | 10.11 | 2.08 |
| XL Batch | 128x4096 | 49.13 | 28.86 | 1.70 | 21.36 | 10.10 | 2.11 |
| Odd Dims | 17x1023 | 58.22 | 40.10 | 1.45 | 24.21 | 10.01 | 2.42 |
| GPT-3 Large | 32x12288 | 54.52 | 29.98 | 1.82 | 24.38 | 12.46 | 1.96 |

`avg_realistic_speedup` 1.8257, `avg_optimal_speedup` 2.1257.

`statistical` (30 samples of a 100-launch `perf_counter` loop each, `ttest_ind`):

| config | ratio of means | ratio std | p-value |
|---|---|---|---|
| BERT (32x768) | 2.09 | 0.03 | 1.32e-86 |
| GPT-3 (32x4096) | 2.08 | 0.03 | 1.43e-94 |
| Odd (17x1023) | 2.42 | 0.22 | 2.47e-44 |

`edge_cases` (`torch.allclose(rtol=1e-4, atol=1e-5)` vs a fresh `nn.LayerNorm`,
plus a 100-launch optimal-mode timing): 1x1, 1x17, 1000x1, 1x32768, 13x13,
1x4095, 1x4097, 512x512 — all `correct: true`; ratios 2.38, 2.41, 2.37, 2.36,
2.31, 2.33, 2.36, 2.10.

`accuracy` (32x4096 fp32, identity affine — weight 1, bias 0 — difference to
PyTorch's output, not error against a reference):

| scenario | max abs diff | mean abs diff | max rel diff |
|---|---|---|---|
| Normal | 4.77e-07 | 2.03e-08 | 1.43e-04 |
| Large Values (x1000) | 4.77e-07 | 9.63e-09 | 2.06e-04 |
| Small Values (x0.001) | 1.19e-07 | 8.48e-10 | 6.44e-04 |
| Near Zero (x1e-6) | 1.16e-10 | 4.49e-13 | 1.08e-04 |
| Mixed Range | 4.77e-07 | 1.42e-08 | 4.86e-05 |

`bandwidth`: `kernel_time_ms` 0.04096 (40.96 µs) for one event-timed 32x4096
fp32 call with no warm-up loop of its own (the process itself was warm by then), `bandwidth_gb_s` 26.40, `peak_bandwidth_gb_s` 1555 (hard-coded),
`utilization_percent` 1.70.

## How to read these numbers

* **They are launch/dispatch-bound.** In optimal/cached mode the extension
  costs ~10 µs per call and PyTorch ~21 µs per call (grad disabled,
  `publication_validation_results.json`) or ~26–30 µs per call (autograd on,
  `publication_results.json`) **regardless of size** — 10.06 µs at 1x768
  (3 KB) and 10.10 µs at 128x4096 (2 MB) for the extension; 21.11 µs and
  21.36 µs for PyTorch in the same file. A time that does not move when the
  data grows ~680x is the host's per-launch cost. The extension only leaves
  that floor at the largest shape, 32x12288 (1.5 MB): 12.46 µs. PyTorch is
  higher at 17x1023 (24.21 µs) and at 32x12288 (24.38 µs). For N % 4 != 0
  PyTorch v2.7.1 launches two kernels instead of one (see
  [`../../../docs/methodology.md`](../../../docs/methodology.md) §5); whether
  that explains the 17x1023 number is untested — 32x12288 takes the
  single-kernel path and shows a similar increase.
* **The average ratio (2.13x with grad disabled, 2.57x with autograd on;
  individual configs 1.96x–2.76x) is therefore a ratio of host overheads** — a
  `torch.nn.LayerNorm` call through the dispatcher (with autograd in
  `publication_results.json`) versus a bare pybind call into a kernel — **not
  a kernel speedup**. Nothing in these files says how long either kernel runs
  on the GPU. Kernel time was never measured (see
  [`../../README.md`](../../README.md) for how to measure it now).
* **"Realistic" mode adds launch latency on an idle GPU** (single call between
  event records after a `synchronize`), no warm-up, PyTorch always first, and
  event creation inside PyTorch's timed region; the 400 µs standard deviation
  on the first (BERT) sample set is consistent with an un-warmed first call. See
  [`../../legacy/README.md`](../../legacy/README.md).
* **The bandwidth line uses the wrong peak and a cold call.** Recomputed:
  bytes = (32·4096·2 + 4096·2)·4 = 1,081,344; 1,081,344 B / 40.96 µs =
  26.4 GB/s = **1.3 % of 2039 GB/s** (the A100-SXM4-80GB peak from the NVIDIA
  A100 datasheet), not 1.7 % of 1555 GB/s (the 40 GB part). Using the same
  file's pipelined optimal-mode 9.99 µs at 32x4096 instead would imply
  ~108 GB/s = 5.3 % of peak — still a host-bound number, not a kernel
  measurement.
* **The accuracy block is a discrepancy from PyTorch, not accuracy.** 4.77e-7
  max abs difference at 32x4096 fp32 with identity affine says the two fp32
  implementations agree to within 4.77e-7 on that input; it says nothing about
  correctness with real weights/biases (never tested here) or about float64
  (which the `3686b4c` kernel accumulated in float).

## Numbers in the old README that are NOT in these files

The pre-rewrite README (`git show 12dee09:README.md`) quoted the following;
the table and the averages appear in neither JSON in this folder nor anywhere
else in the repository:

* "1.86x" (realistic) and "2.66x" (optimal) average speedups — the files say
  1.8544/1.8257 and 2.5653/2.1257;
* the five-row per-config table "BERT 2.36x / 2.57x, GPT-2 1.73x / 2.61x,
  GPT-3 1.64x / 2.64x, Large Batch 1.74x / 2.67x, Odd Dimensions 1.81x / 2.83x"
  (two of its ten values do occur in `publication_results.json`, but not for
  those rows: 1.73 is the Large Batch realistic ratio there and 2.57 is the
  cached average);
* "167 GB/s" actual bandwidth and "10.7 %" utilisation — the only committed
  bandwidth figure is 26.4 GB/s / 1.70 % (of 1555) above.

Where those figures came from is not recorded in the repository history.
