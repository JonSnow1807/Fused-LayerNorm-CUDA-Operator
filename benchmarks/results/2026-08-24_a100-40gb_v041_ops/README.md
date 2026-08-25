# 2026-08-24 — A100-SXM4-40GB — v0.4.1 op-family measurements

> **Superseded by `../2026-08-25_a100-40gb_v042_ops/`** (v0.4.2 fp8
> NaN-scale fix + locked-clock methodology). Caveat when comparing: this run
> used default, unlocked clocks — its small-shape (~10 µs) kernel times
> reflect a favourable clock state that did not reproduce the next day
> (up to 29 % on `rms_norm_fp8` at 512×1024; `docs/methodology.md` §7).
> Large-shape numbers reproduce within ~2 %.

Supersedes `../2026-08-24_a100-40gb_v040_ops/` (kept as the historical record
of the v0.4.0 claims) for two reasons: v0.4.1's benchmark **fairness fix** —
the eager fp8 composite previously computed `y.float()` twice, handicapping
the competitor, so every fp8-vs-eager ratio changed — and the v0.4.1 kernel
changes (`__restrict__` removal on the aliasable residual pair, fp8 NaN
passthrough), which re-validate the rest within noise.

Both files produced by `benchmarks/bench_norms.py` from a **clean clone of the
v0.4.1 code commit `1193a53`** (JSON metadata: `git_dirty = false`,
`extension_version = 0.4.1`, driver 570.148.08, `TORCH_CUDA_ARCH_LIST = 8.0`,
PyTorch 2.13.0+cu129 / CUDA 12.9); the 257-test suite passed in the same
clone first. Idle GPU, sequential runs. Method and candidate set as in the
v0.4.0 README (correctness gates — now with an independent dynamic-scale
check — then eager latency, profiler kernel time, CUDA-graph replay; the
`*_bwd` rows time a full forward+backward training step).

Summary (kernel time, ours vs competitor; ranges over the six shapes,
512×1024 … 4096×8192):

| op (fp16) | vs eager composite | vs torch.compile'd composite | peak-BW* |
|---|---|---|---|
| `fused_add_rms_norm` | 1.23–1.58× | 0.80–0.99× (0.94–0.99× at M ≥ 2048) | 62–85 % |
| `fused_add_layer_norm` | 1.09–1.49× | 0.52–1.00× (0.88–1.00× at M ≥ 2048) | 27–85 % |
| `rms_norm` vs `F.rms_norm` | 0.99–1.27× | 0.77–1.24× | 33–116 %* |
| `rms_norm_fp8` (dynamic) | 6.30–7.19× | **1.06–1.77× — ≥ 1 at every shape** | 13–40 % |
| `fused_add_rms_norm_fp8` (dyn.) | 5.20–6.52× | **1.18–1.79× — ≥ 1 at every shape** | 29–68 % |
| training step fwd+bwd (LN / RMS) | 0.44–1.04× / 0.55–1.09× | — | — |

fp32: `fused_add_rms_norm` 1.24–1.68× vs eager (0.85–1.04× vs compiled);
`fused_add_layer_norm` 1.27–1.43× (0.59–1.03×); `rms_norm` 1.02–1.30× vs
aten; fp8 ops 2.93–6.61× vs eager, 1.09–1.64× vs compiled; training step
0.54–1.23× (LN) / 0.64–1.28× (RMS) — the fp32 training step beats autograd at
the largest shape.

Wall clock (`eager_us`), fused-add ops: faster than the eager composite at
every measured shape and dtype (1.14–1.49×); vs the compiled composite,
faster or tied everywhere in fp16 (ties 0.99–1.01× at the two largest
shapes; up to 4.98× at 512×1024) and in fp32 faster or tied except the
largest shape, where the compiled composite wins ~15 % (0.85–0.87× at
4096×8192). At 512×1024 fp16 the compiled candidates pay 4.4–10.7× more wall
clock than these ops, depending on the op.

\*A few rows exceed 100 % of the 1 555 GB/s datasheet peak: the bytes models
count each tensor once, and at those shapes the 8–34 MB working set stays
resident in the A100's 40 MB L2 across the 200 timed calls, so much of the
modelled traffic never reaches DRAM. Compiled candidates show the same
effect; candidate ratios are unaffected.

The training-step rows remain the deliberately-published weak spot: the
backward kernels are correctness-first (scalar loads, deterministic
no-atomics parameter grads, gradcheck-verified); fp16 measures 0.44–1.09× of
PyTorch's autograd. Training through these ops is correct and bitwise
reproducible; making it fast is future work.
