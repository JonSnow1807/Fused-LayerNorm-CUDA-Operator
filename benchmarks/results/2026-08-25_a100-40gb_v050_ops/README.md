# 2026-08-25 — A100-SXM4-40GB — v0.5.0 op-family measurements

Supersedes `../2026-08-25_a100-40gb_v042_ops/` (kept as history). What
changed in v0.5.0: the backward kernels were vectorised (16-byte loads, a
256-thread partials block, a GPU-filling chunk policy and a single fused
stage-2 finalize kernel — the training-step rows are where to look);
`layer_norm_gelu` gained a real backward (new `layer_norm_gelu_bwd` row);
and the LayerNorm family gained fp8 outputs (new `layer_norm_fp8_dynamic`
and `fused_add_layer_norm_fp8_dynamic` rows). Forward rows for the
pre-existing ops re-validate the v0.4.2 numbers within noise.

Both files produced by `benchmarks/bench_norms.py` from a **clean clone of
`b38a935`** (JSON metadata: `git_dirty = false`, `extension_version =
0.5.0`, driver 570.148.08, `gpu_clock_state` showing SM clocks locked at
1410 MHz with persistence on — methodology §7). Extension built in the
clone with `TORCH_CUDA_ARCH_LIST=8.0`; PyTorch 2.13.0+cu129 / CUDA 12.9.
The 289-test suite passed in the same clone in all three kernel modes,
plus the randomized contract fuzz. Idle GPU, sequential runs. Method and
candidate set as in the v0.4.2 README; the `*_bwd` rows time a full
forward+backward training step through the custom-op dispatcher and the
autograd engine (both sides pay their own engine overhead).

## The headline change: training steps (kernel time, ours vs PyTorch autograd)

| training step | fp16 | fp32 |
|---|---|---|
| `layer_norm` fwd+bwd | **0.75–1.42×** (1.20–1.42× at M ≥ 2048) | 0.83–1.13× |
| `rms_norm` fwd+bwd | **0.82–1.44×** (1.17–1.44× at M ≥ 2048) | 0.86–1.16× |
| `layer_norm_gelu` fwd+bwd | 0.80–1.44× | **0.95–1.88×** |

v0.4.2 measured 0.43–1.10×; the backward is no longer the published weak
spot at production shapes — the remaining sub-1× cell is 512×1024, where
both sides are engine/launch-bound and ours pays more launches. Wall-clock
training-step ratios at small M remain dispatch-dominated (0.45–0.6× at
512×1024) — same story and same remedies (torch.compile, CUDA graphs) as
the forward's dispatcher overhead, and unchanged by this release. Backward
determinism (no atomics, fixed chunk grid) is unchanged and still tested;
dgamma/dbeta bit values differ from v0.4.2 (different reduce split/order),
run-to-run bitwise reproducibility holds.

## Forward summary (kernel time, ours vs competitor; six shapes, 512×1024 … 4096×8192)

| op (fp16) | vs eager composite | vs torch.compile'd composite | peak-BW* |
|---|---|---|---|
| `fused_add_rms_norm` | 1.23–1.59× | 0.81–0.99× (0.96–0.99× at M ≥ 2048) | 64–86 % |
| `fused_add_layer_norm` | 1.08–1.49× | 0.52–1.00× (0.88–1.00× at M ≥ 2048) | 35–84 % |
| `rms_norm` vs `F.rms_norm` | 0.99–1.29× | 0.77–1.26× | 35–121 %* |
| `rms_norm_fp8` (dynamic) | 4.93–7.23× | **1.04–1.76× — ≥ 1 at every shape** | 13–40 % |
| `fused_add_rms_norm_fp8` (dyn.) | 4.96–5.99× | **1.01–1.76× — ≥ 1 at every shape** | 29–68 % |
| `layer_norm_fp8` (dynamic) | 3.47–5.97× | 1.02–1.49× at M ≥ 2048 (0.75× at 512×1024) | 9–33 % |
| `fused_add_layer_norm_fp8` (dyn.) | 3.56–5.66× | 1.25–1.68× at M ≥ 2048 (0.74× at 512×1024) | 20–64 % |

fp32: `fused_add_rms_norm` 1.25–1.70× vs eager (0.85–1.04× vs compiled);
`fused_add_layer_norm` 1.27–1.43× (0.67–1.02×); `rms_norm` 1.02–1.31× vs
aten; RMS fp8 ops 2.98–5.96× vs eager and **1.10–1.69× vs compiled (≥ 1 at
every shape)**; LN fp8 ops 2.91–5.11× vs eager, 1.06–1.48× vs compiled at
M ≥ 2048 (0.87–0.88× at 512×1024).

The "≥ 1× at every shape" claim is deliberately RMS-only: the new LN fp8
ops lose the smallest shape to Inductor's fused small-shape kernel and the
tables say so. On wall clock every fp8 op beats the compiled chain at every
measured shape and dtype (1.08–10.78×) — the compiled candidates pay their
guard/dispatch cost per eager call.

Wall clock (`eager_us`), fused-add ops: faster than the eager composite at
every measured shape and dtype (1.13–1.48×); vs the compiled composite,
faster or tied in fp16 except one 0.99× tie-breaker at 4096×4096
(up to 4.55× at 512×1024) and in fp32 faster or tied except the largest
shape, where the compiled composite wins ~15 % (0.85–0.86× at 4096×8192).

\*A few rows exceed 100 % of the 1 555 GB/s datasheet peak: the bytes
models count each tensor once, and at those shapes the 8–34 MB working set
stays resident in the A100's 40 MB L2 across the 200 timed calls, so much
of the modelled traffic never reaches DRAM. Compiled candidates show the
same effect; candidate ratios are unaffected.
