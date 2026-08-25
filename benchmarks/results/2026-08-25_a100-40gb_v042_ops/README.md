# 2026-08-25 — A100-SXM4-40GB — v0.4.2 op-family measurements

Supersedes `../2026-08-24_a100-40gb_v041_ops/` (kept as history) for two
reasons: the v0.4.2 **fp8 NaN-scale fix** — the dynamic-scale amax is now
NaN-propagating, which costs the two fp8-dynamic ops ~2–4 % kernel time and
re-measures every fp8 claim — and a **methodology change: SM clocks are now
locked** (`nvidia-smi -pm 1; nvidia-smi -lgc 1410,1410`) before every run.
The clock lock exists because re-measuring v0.4.1 a day later showed its
small-shape kernel times were up to 29 % slower under that day's default
clock state (1095 MHz application clocks vs 1410 MHz boost; short profiler
loops don't reliably ramp), while ≥ 100 µs kernels reproduced within ~2 % —
see `docs/methodology.md` §7. Locked clocks reproduce the v0.4.1 committed
numbers, so the two directories differ mainly where the fix and the clock
discipline say they should. Do not compare the smallest shapes across the
two directories.

Both files produced by `benchmarks/bench_norms.py` from a **clean clone of
`c24b55a`** (the kernel-change commit `737b831` plus a benchmark-metadata
commit that records the clock state). JSON metadata: `git_dirty = false`,
`extension_version = 0.4.2`, driver 570.148.08, and a `gpu_clock_state`
field showing SM clocks pinned at 1410 MHz with persistence mode on —
committed data now *proves* its clock state instead of asserting it. The
extension was built in the clone with `TORCH_CUDA_ARCH_LIST=8.0`; PyTorch
2.13.0+cu129 / CUDA 12.9. The 260-test suite passed in the same clone first
(all three kernel modes), plus a 130-config randomized contract fuzz on the
release build — the fuzz is what found the NaN-scale bug this release
fixes. Idle GPU, sequential runs. Method and candidate set
as in the v0.4.1 README (correctness gates with an independent dynamic-scale
check, then eager latency, profiler kernel time, CUDA-graph replay; the
`*_bwd` rows time a full forward+backward training step).

Summary (kernel time, ours vs competitor; ranges over the six shapes,
512×1024 … 4096×8192):

| op (fp16) | vs eager composite | vs torch.compile'd composite | peak-BW* |
|---|---|---|---|
| `fused_add_rms_norm` | 1.23–1.58× | 0.80–0.99× (0.96–0.99× at M ≥ 2048) | 63–85 % |
| `fused_add_layer_norm` | 1.08–1.49× | 0.52–1.00× (0.88–1.00× at M ≥ 2048) | 35–84 % |
| `rms_norm` vs `F.rms_norm` | 0.98–1.29× | 0.77–1.28× | 34–117 %* |
| `rms_norm_fp8` (dynamic) | 4.82–7.04× | **1.03–1.73× — ≥ 1 at every shape** | 13–39 % |
| `fused_add_rms_norm_fp8` (dyn.) | 4.93–5.92× | **1.01–1.74× — ≥ 1 at every shape** | 29–67 % |
| training step fwd+bwd (LN / RMS) | 0.43–1.04× / 0.55–1.10× | — | — |

fp32: `fused_add_rms_norm` 1.24–1.70× vs eager (0.84–1.04× vs compiled);
`fused_add_layer_norm` 1.27–1.43× (0.67–1.02×); `rms_norm` 1.02–1.32× vs
aten; fp8 ops 2.93–5.80× vs eager, 1.09–1.67× vs compiled; training step
0.53–1.24× (LN) / 0.64–1.28× (RMS) — the fp32 training step beats autograd
at the three largest shapes.

The fp8 "≥ 1× vs compiled at every shape" headline survives the NaN fix, but
its 512×1024 fp16 margin is now 1.01–1.03× — three points went to NaN
correctness, and at that scale the number is best read as parity. The
fp8-vs-eager ratios are lower than the v0.4.1 directory's not because the
kernels regressed but because locked clocks help the competitor most: the
eager chain is many short kernels, exactly the shape of work that was
under-clocked before.

Wall clock (`eager_us`), fused-add ops: faster than the eager composite at
every measured shape and dtype (1.12–1.48×); vs the compiled composite,
faster or tied everywhere in fp16 (ties 0.99–1.01× at the two largest
shapes; up to 4.60× at 512×1024) and in fp32 faster or tied except the
largest shape, where the compiled composite wins ~15 % (0.85–0.86× at
4096×8192). At 512×1024 the compiled candidates pay 4.1–10.4× more wall
clock than these ops, depending on the op and dtype.

\*A few rows exceed 100 % of the 1 555 GB/s datasheet peak: the bytes models
count each tensor once, and at those shapes the 8–34 MB working set stays
resident in the A100's 40 MB L2 across the 200 timed calls, so much of the
modelled traffic never reaches DRAM. Compiled candidates show the same
effect; candidate ratios are unaffected.

The training-step rows remain the deliberately-published weak spot: the
backward kernels are correctness-first (scalar loads, deterministic
no-atomics parameter grads, gradcheck-verified); fp16 measures 0.43–1.10× of
PyTorch's autograd (fp32 0.53–1.28×, ahead at the three largest shapes).
Training through these ops is correct and bitwise reproducible; making it
fast is future work.
