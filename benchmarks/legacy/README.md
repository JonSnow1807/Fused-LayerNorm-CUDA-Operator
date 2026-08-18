# Legacy benchmark scripts (August 2025) — kept for provenance, do not use

The three scripts in this directory are the scripts behind the speedup claims
in the pre-rewrite README (`git show 12dee09:README.md`); note that their
committed outputs do not contain the exact figures that README quoted — see
[`../results/2025-08-17_a100_eager_latency/README.md`](../results/2025-08-17_a100_eager_latency/README.md).
They are kept
**byte-identical** to their last committed versions so that anyone can trace
where a published number came from. They are not maintained, they depend on
`numpy`/`scipy` (not project dependencies any more), they call the extension
with the *old* positional signature `layernorm(x, weight, bias, 1e-5)`, and
their methodology has the problems listed at the bottom. Use
[`../bench_layernorm.py`](../bench_layernorm.py) instead.

Original locations (moved here in the rewrite): `benchmarks/reproduce_speedup.py`,
`tests/publication_ready_benchmark.py`, `tests/publication_validation.py`.

## What each script does

### `reproduce_speedup.py` (added in commit `4ea14d2`, rewritten to its current form in `cc10eea`, both 2025-08-17)

The script the old README told users to run ("Run
`python benchmarks/reproduce_speedup.py` to verify these results on your
hardware.").
**It writes no file**; it only prints. There is no committed artifact from it.

* `benchmark_realistic` (lines 14–44): for each of 100 samples, a fresh
  `torch.randn` input (line 22); `torch.cuda.synchronize()`, two new
  `torch.cuda.Event`s, `start.record()`, **one** call of the `nn.LayerNorm`
  module, `end.record()`, synchronize (lines 25–32); then the same event pair
  around one call of `fused_layernorm_cuda.layernorm` (lines 35–39). Reports the
  mean of the 100 event-pair times per implementation and their ratio.
* `benchmark_optimal` (lines 46–77): one input tensor reused; 200 warm-up calls
  of both (lines 52–54); then one event pair around a loop of 1000 launches of
  the module (lines 59–66) and one around 1000 launches of the extension
  (lines 69–74); reports elapsed / 1000.
* Five configs (lines 80–86: 32x768, 32x1024, 32x4096, 64x4096, 17x1023), then
  a "not hardcoded" demonstration that reruns the 32x768 optimal case five times
  (lines 132–134).
* Autograd is never disabled; `nn.LayerNorm(hidden).cuda()` (line 16/49) has
  parameters that require grad, so every timed `ln(x)` call records an autograd
  graph while the extension call does not.

### `publication_ready_benchmark.py` (added in commit `4ea14d2`, 2025-08-17)

Produced
[`../results/2025-08-17_a100_eager_latency/publication_results.json`](../results/2025-08-17_a100_eager_latency/publication_results.json)
(written by lines 133–143 as `publication_results.json` in the working
directory; the kernel it ran was the one from commit `3686b4c`).

* `benchmark_realistic` (lines 14–48): identical method to the one above
  (fresh tensor, event pair around one Python call, 100 samples, PyTorch first).
* `benchmark_cached` (lines 50–84): identical to `benchmark_optimal` above
  (200 warm-ups, event pair around 1000 launches).
* Same five configs (lines 86–92). Output keys: `realistic` (mean/std per
  config), `cached` (per-call time per config), `summary` (mean of the five
  ratios in each mode).
* Autograd is never disabled (no `torch.no_grad()` / `set_grad_enabled` in the
  file).

### `publication_validation.py` (added in commit `12dee09`, 2025-08-17)

Produced
[`../results/2025-08-17_a100_eager_latency/publication_validation_results.json`](../results/2025-08-17_a100_eager_latency/publication_validation_results.json)
(written by `save_results`, lines 474–505). Its `metadata` block records
torch 2.7.1+cu128, CUDA 12.8, NVIDIA A100-SXM4-80GB, timestamp
2025-08-17T20:00:04.

* `torch.set_grad_enabled(False)` at line 22 — this is the only one of the
  three scripts that disables autograd for the module call.
* `test_performance` (lines 66–139): ten configs (lines 70–81);
  `benchmark_realistic` (lines 141–167, 50 samples, same single-call event-pair
  method as above) and `benchmark_optimal` (lines 169–199, 100 warm-ups, event
  pair around 500 launches).
* `test_numerical_accuracy` (lines 201–253): compares the extension to a fresh
  `nn.LayerNorm` — i.e. weight = 1, bias = 0 — on 32x4096 fp32 inputs at
  several scales; reports max/mean absolute and max relative *difference from
  PyTorch* (not error against a higher-precision reference).
* `test_statistical_significance` (lines 255–316): 30 samples per
  implementation, each sample = `time.perf_counter()` around a loop of 100
  launches + synchronize (lines 277–292); `scipy.stats.ttest_ind` on the two
  sample sets (line 300).
* `test_edge_cases` (lines 318–365): `torch.allclose(rtol=1e-4, atol=1e-5)`
  against a fresh (identity-affine) `nn.LayerNorm`, plus a 100-iteration
  optimal-mode timing per case.
* `test_memory_bandwidth` (lines 367–418): **one** event pair around **one**
  extension call on 32x4096 fp32 with no immediately preceding warm-up
  (lines 376–384); bytes = (M·N·2 + N·2)·4 (line 390); peak hard-coded to
  1555 GB/s for any device name containing "A100" (lines 396–397) and to
  1000 GB/s for anything unrecognised (line 401).

## Known methodological problems (facts, with evidence)

1. **Both modes measure host dispatch + launch cost, not kernel time.** In
   `publication_validation_results.json` → `performance.optimal`, the
   extension's per-call time is 10.06 µs at 1x768 fp32 (3 KB of input) and
   10.10 µs at 128x4096 (2 MB of input); PyTorch's is 21.11 µs and 21.36 µs.
   A number that does not change when the data grows ~680x is the host's
   per-launch cost, not the kernel's. The "realistic" mode is worse: a single
   event pair around one Python call on an idle GPU measures launch latency
   plus the call itself (see [`../../docs/methodology.md`](../../docs/methodology.md)).
2. **Unequal baselines: `nn.LayerNorm` was called with autograd enabled** in
   `reproduce_speedup.py` and `publication_ready_benchmark.py`, so PyTorch's
   timed call also builds an autograd node and saves `mean`/`rstd`, while the
   extension call is a bare kernel launch. `publication_validation.py` disabled
   grad (line 22) and its optimal-mode average ratio is lower: 2.13x
   (`avg_optimal_speedup` = 2.1256…) versus 2.57x (`summary.cached_speedup` =
   2.5652… in `publication_results.json`).
3. **"Realistic" mode has no warm-up, always times PyTorch first, and creates
   CUDA events inside PyTorch's timed region.** `torch.cuda.Event` allocates
   the underlying CUDA event lazily on its first `record()`; in each sample the
   two events are new, so the `end` event is created between `start.record()`
   and `end.record()` of the *PyTorch* measurement, whereas the extension call
   that follows re-uses the already-created events. The first sample of the
   first config also carries process cold-start: `publication_results.json`
   `realistic[0].pytorch_std` = 0.4000 ms (400 µs) against a mean of 0.1009 ms.
4. **The "cached vs. uncached" (optimal vs. realistic) explanation is wrong.**
   In both modes the input was produced on the GPU by `torch.randn` immediately
   before it was read, so it is equally likely to be L2-resident either way
   (the largest input, 32x12288 fp32, is 1.5 MB). The two modes differ in
   *timing method* — one synchronised call versus a pipelined loop of
   launches — not in cache state.
5. **The bandwidth figure hard-codes the wrong peak and times one cold call.**
   `publication_validation.py` uses 1555 GB/s for anything called "A100"
   (line 397), but the run was on an A100-SXM4-80GB, whose peak is 2039 GB/s
   (NVIDIA A100 datasheet). The measured 40.96 µs (`bandwidth.kernel_time_ms`
   = 0.04096) is a single, un-warmed, event-pair call; the same script's own
   pipelined optimal-mode number for that shape is 9.99 µs. The resulting
   26.4 GB/s is 1.3 % of 2039 GB/s (the file says 1.7 % of 1555).
6. **The t-tests are meaningless for the claim being made.** They compare two
   sets of near-constant host overheads (30 samples of a 100-launch loop each);
   with sample standard deviations of a few percent any difference in means is
   "significant". The p-values (1.32e-86, 1.43e-94 and 2.47e-44 for the three
   configs) say nothing about kernel speed.
7. **All correctness checks used identity affine parameters** (a freshly
   constructed `nn.LayerNorm`, weight = 1, bias = 0), so a kernel that ignored
   `weight`/`bias` entirely would have passed. The new tests use random
   non-identity affine parameters.

Nothing in these scripts should be run to "verify" the current kernel; run
`python benchmarks/bench_layernorm.py` and read
[`../../docs/methodology.md`](../../docs/methodology.md).
