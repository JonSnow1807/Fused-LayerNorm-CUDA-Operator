# 2026-10-29 — H100 NVL — v0.6.0 Hopper RMSNorm fast paths (branch `5268275`) vs v0.5.0 (`origin/main` `93d5882`)

The first A/B of two builds in this repository, and the third GPU. Both
halves were produced by `benchmarks/bench_norms.py` on the same machine, the
same afternoon, with the same arguments: the PR #2 branch
(`h200-rmsnorm-fp8-fast-paths` at `5268275`, "Hopper RMSNorm fast paths") in
`branch_5268275/`, the baseline (`origin/main` at `93d5882`) in
`main_93d5882/`. Every ratio below is **branch kernel time / baseline kernel
time — below 1 means the branch is faster**. The PR's own performance numbers
were taken on an H200; this directory re-measures the design on an H100 NVL
with a different toolchain, so only the ratios are comparable to the PR's
claims, never the absolute microseconds.

## Files

| path | what |
|---|---|
| `branch_5268275/bench_norms_2026-10-29_22{2015,2203,2353}_h100-nvl_{float16,bfloat16,float32}.{json,md}` | PR #2 branch at `5268275` (`metadata.git_commit`); `norm_fwd_rms_rowreg_kernel` and `norm_fwd_rms_fp8dyn_warp_kernel` appear in `kernel_names` where the new paths dispatch |
| `main_93d5882/bench_norms_2026-10-29_22{3322,3442,3602}_h100-nvl_{float16,bfloat16,float32}.{json,md}` | baseline, `origin/main` at `93d5882` |
| `kernel_ratio_table.md` | derived: one row per (op, dtype, shape) with both builds' kernel names, kernel and graph-replay times and the two ratios, generated from the twelve files above by a throw-away script (not a script of this repository) |

Baseline footnote: tag `v0.5.0` is `9a74cf7`; `origin/main` `93d5882` is two
commits past it — `7e8569e` (H100 validation data: docs and result files
only) and `93d5882` (reviewer-pass cleanup: comment-only edits in the
forward kernel sources, one shared-memory tile change in the *backward*
partials kernel, dead Python removed). The forward kernels timed here are
source-identical to the tag, which is why the baseline is called "v0.5.0"
below.

## Environment and provenance

* Vast.ai Docker container; **NVIDIA H100 NVL** (95 GB, PCIe form factor,
  132 SMs, 400 W power cap — not the H100 80GB HBM3 SXM of
  `../2026-08-25_h100-80gb_v050_ops/`), driver 580.173.02, AMD EPYC 9V84
  host, Linux 7.0.0-1017-azure, Python 3.12.3. Extension built in each
  checkout with `pip install --no-build-isolation -e .` against
  torch 2.13.0+cu130 (CUDA runtime 13.0; `nvcc` 13.4, V13.4.59) with
  `TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX"` as recorded in
  `metadata.env_torch_cuda_arch_list`. This is a different toolchain from
  the PR's H200 measurements (cu129 / nvcc 12.9): a re-measurement of the
  design, not of the identical binaries.
* **Clocks could not be locked** (no permission for `nvidia-smi -lgc` in
  the container; methodology §7). `metadata.gpu_clock_state.clocks.sm` reads
  1785 MHz — the device's max and application clock — at the start of every
  one of the six runs; the idle reading before and after the session was
  345 MHz; there is no in-run sampling. Both builds ran under the same
  conditions, and the 70 control rows (same kernel in both builds, below)
  bound the drift: median 1.000, extremes 0.961–1.045. Read the absolute
  small-shape microseconds as a snapshot and the ratios as the result.
* `metadata.git_dirty = true` in all six JSONs, against the policy in
  `../../README.md`. The scripts mark the tree dirty on any
  `git status --porcelain` output, and each run wrote its output into an
  untracked `benchmarks/results/2026-10-29_h100-nvl_*` directory inside the
  checkout it benchmarked. Consistent with that, an earlier (uncommitted)
  `bench_layernorm.py` session in each checkout recorded `git_dirty = false`
  for its first file and `true` for the following ones. No `git status` was
  captured, so this is the documented explanation, not a verified one; the
  commits themselves are `5268275` and `93d5882` exactly
  (`metadata.git_commit`).
* Both builds report `extension_version = 0.5.0` (the branch was bumped
  after it was measured); the builds are told apart by `metadata.git_commit`
  and by the kernel names in `kernel_names`.
* Not interleaved: the three branch runs span 22:18–22:24 UTC, the three
  baseline runs 22:32–22:36 UTC (`metadata.timestamp`). A first baseline
  session at 22:24–22:28 UTC was discarded: its JSONs carried commit
  `93d5882` but the editable install had not switched extensions, so their
  `kernel_names` still showed the branch's kernels. The committed baseline
  half was re-run after `pip uninstall` + `pip install --force-reinstall -e`,
  with the loaded extension's path asserted at the start of each process.
  An earlier `bench_layernorm.py` pair (plain LayerNorm, untouched by the
  release) is not committed for the same install-provenance reason.
* Same commit, same machine, earlier the same day: 335/335 tests in each of
  the default / scalar / vec kernel modes, `tests/fuzz_contracts.py` ALL
  PASS in each, compute-sanitizer racecheck 0 hazards on the edge-shape
  tests, memcheck with no kernel-side report (its two reported "errors" are
  the container's `cudaGetDeviceCount` IMEX-channel API messages at import).
  Those logs are not committed.

## Command

`bench_norms.py` compiles one `torch.compile` candidate per shape and trips
Dynamo's recompile limit at 18 shapes (`FailOnRecompileLimitHit` under
`fullgraph=True`), so each run called `main()` in-process with the limit
raised. That touches only the `torch.compile(composite)` candidate, not the
library candidate the ratios use. Per dtype, from the checkout being
measured (`$REPO` = `/root/fln` for the branch, `/root/fln-main` for the
baseline; `$OUT` = a directory under that checkout's `benchmarks/results/`):

```bash
OPS="--op rms_norm --op fused_add_rms_norm --op rms_norm_fp8_dynamic --op fused_add_rms_norm_fp8_dynamic"
SHAPES="32x768,32x1024,32x4096,17x1023,128x4096,512x1024,1x4096,16x4096,64x4096,192x4096,200x4096,255x4096,256x4096,257x4096,64x8192,32x16384,4096x4096,8192x1024"
python - "$REPO" "$OUT" "$DTYPE" $OPS --shapes "$SHAPES" <<'PY'
import sys, torch, torch._dynamo
torch._dynamo.config.cache_size_limit = 10000
for a in ("recompile_limit", "accumulated_recompile_limit"):
    try: setattr(torch._dynamo.config, a, 100000)
    except Exception: pass
repo, out, dt, *rest = sys.argv[1:]
sys.path.insert(0, repo + "/benchmarks"); import bench_norms
sys.exit(bench_norms.main(rest + ["--dtype", dt, "--out", out]))
PY
```

(The branch half set `recompile_limit` to 10 000 instead of 100 000; both
are far above the 18 recompiles needed.) `--iters 200`, `--reps 20`,
`--warmup 50` — the defaults; `metadata.argv` records the full argument
list. `kernel us` = `torch.profiler` CUDA kernel time summed over 200 calls
after 50 warm-up calls, divided by 200; `graph us` = median of 20 replays of
one CUDA graph of 200 calls, with min and max recorded. The shape list is
`bench_layernorm.py`'s default grid plus the PR's few-row shapes and
4096×4096 / 8192×1024 — not `bench_norms.py`'s own default grid. Four
RMSNorm ops × 18 shapes × 3 dtypes = 216 library rows per build. The peak
bandwidth is unknown for this device name, so the `% peak` columns read
`n/a`; nothing below depends on it.

## Which rows run which kernel

Classified from `kernel_names` of the library candidate in the two JSONs of
each row:

| group | rows | branch kernel | baseline kernel |
|---|---|---|---|
| **few-row kernel** (PR change 2): M < 256, 16-byte-aligned rows, N ≤ 16384 fp16/bf16 or ≤ 8192 fp32 — 12 shapes per op and dtype, 11 in fp32 (32×16384 is above the fp32 limit) | 140 | `norm_fwd_rms_rowreg_kernel` | `norm_fwd_kernel` / `norm_fwd_vec_kernel` |
| **warp-per-row dynamic-FP8 kernel** (change 1): `rms_norm_fp8_dynamic` at 512×1024 and 8192×1024, `fused_add_rms_norm_fp8_dynamic` at 8192×1024; fp16 and bf16 | 6 | `norm_fwd_rms_fp8dyn_warp_kernel` | `norm_fwd_vec_kernel` |
| **control**, same kernel in both builds: every M ≥ 256 row except the six above, 17×1023 (rows not 16-byte aligned), 32×16384 fp32, the fp32 fp8-dynamic rows at N = 1024 (outside the path's fp32 range), `fused_add_rms_norm_fp8_dynamic` at 512×1024 (the branch keeps it on the vectorised kernel) | 70 | identical | identical |

Change 1 was therefore measured only at hidden size 1024: the grid has no
shape with M ≥ 256 and N ≤ 512, and the path's fp32 range ends at N = 512.
**The PR's claims for this path at N ≤ 512 and in fp32 rest on its H200 data
alone.**

## Results: branch / v0.5.0 kernel time (below 1 = branch faster)

Pooled over every row of the group, min / median / max; the CUDA-graph
replay ratio alongside:

| group | dtype | rows | kernel ratio | graph ratio |
|---|---|---|---|---|
| few-row kernel | fp16 | 48 | 0.402 / 0.647 / 0.998 | 0.443 / 0.686 / 1.009 |
| few-row kernel | bf16 | 48 | 0.421 / 0.672 / 1.023 | 0.462 / 0.707 / 1.025 |
| few-row kernel | fp32 | 44 | 0.535 / 0.737 / 0.958 | 0.571 / 0.791 / 0.980 |
| warp-per-row fp8 | fp16 | 3 | 0.697 / 0.891 / 0.948 | 0.694 / 0.912 / 0.920 |
| warp-per-row fp8 | bf16 | 3 | 0.672 / 0.896 / 0.909 | 0.672 / 0.890 / 0.917 |
| warp-per-row fp8 | fp32 | 0 | not measured | — |
| control (same kernel) | all | 70 | 0.961 / 1.000 / 1.045 | 0.977 / 1.000 / 1.067 |

Per op on the few-row kernel (kernel ratio min / median / max; graph-replay
median in parentheses):

| op | fp16 (12 rows) | bf16 (12) | fp32 (11) |
|---|---|---|---|
| `rms_norm` | 0.402 / 0.664 / 0.998 (0.729) | 0.421 / 0.699 / 1.023 (0.753) | 0.611 / 0.736 / 0.932 (0.803) |
| `fused_add_rms_norm` | 0.440 / 0.622 / 0.979 (0.671) | 0.480 / 0.640 / 0.989 (0.697) | 0.744 / 0.846 / 0.958 (0.897) |
| `rms_norm_fp8_dynamic` | 0.416 / 0.647 / 0.898 (0.690) | 0.447 / 0.672 / 0.917 (0.712) | 0.535 / 0.649 / 0.872 (0.679) |
| `fused_add_rms_norm_fp8_dynamic` | 0.460 / 0.634 / 0.911 (0.667) | 0.504 / 0.659 / 0.936 (0.692) | 0.636 / 0.732 / 0.889 (0.743) |

All six warp-per-row fp8 rows (kernel µs baseline → branch, ratio; graph µs
the same way):

| op | dtype | M×N | kernel | graph |
|---|---|---|---|---|
| `rms_norm_fp8_dynamic` | fp16 | 512×1024 | 3.70 → 3.29 (0.891) | 4.09 → 3.73 (0.912) |
| `rms_norm_fp8_dynamic` | bf16 | 512×1024 | 3.82 → 3.42 (0.896) | 4.21 → 3.86 (0.917) |
| `rms_norm_fp8_dynamic` | fp16 | 8192×1024 | 15.64 → 10.90 (0.697) | 15.89 → 11.04 (0.694) |
| `rms_norm_fp8_dynamic` | bf16 | 8192×1024 | 16.81 → 11.30 (0.672) | 17.02 → 11.44 (0.672) |
| `fused_add_rms_norm_fp8_dynamic` | fp16 | 8192×1024 | 23.62 → 22.39 (0.948) | 24.50 → 22.53 (0.920) |
| `fused_add_rms_norm_fp8_dynamic` | bf16 | 8192×1024 | 24.66 → 22.42 (0.909) | 25.43 → 22.64 (0.890) |

Example few-row rows and two reference rows (kernel µs baseline → branch;
graph median [min–max]):

| op | dtype | M×N | kernel | graph |
|---|---|---|---|---|
| `rms_norm` | fp16 | 32×16384 | 8.57 → 3.45 (0.402) | 9.11 [9.09–9.15] → 4.03 [4.02–4.08] (0.443) |
| `rms_norm` | bf16 | 32×16384 | 8.59 → 3.62 (0.421) | 9.09 [9.08–9.13] → 4.20 [4.19–4.28] (0.462) |
| `fused_add_rms_norm` | bf16 | 32×16384 | 9.80 → 4.70 (0.480) | 10.28 [10.26–10.33] → 5.27 [5.26–5.31] (0.513) |
| `rms_norm` | fp16 | 64×4096 | 3.43 → 2.26 (0.659) | 3.94 [3.93–4.01] → 2.82 [2.81–2.91] (0.715) |
| `rms_norm` | bf16 | 64×4096 | 3.41 → 2.37 (0.696) | 3.93 [3.92–4.01] → 2.92 [2.92–2.96] (0.744) |
| `rms_norm` | fp32 | 64×4096 | 3.29 → 2.35 (0.715) | 3.79 [3.77–3.82] → 3.02 [3.01–3.04] (0.797) |
| `rms_norm` | bf16 | 32×768 | 1.84 → 1.88 (1.023) | 2.24 [2.23–2.29] → 2.29 [2.28–2.33] (1.025) |
| `rms_norm` | bf16 | 4096×4096 (control) | 21.55 → 21.56 (1.000) | 21.77 [21.68–22.05] → 21.81 [21.66–22.09] (1.002) |

## What the numbers do and do not show

* **The few-row kernel is faster wherever it dispatches on rows of 4096
  elements or more, in all three dtypes** (116 rows, worst 0.924): per-op
  medians 0.62–0.66 (fp16), 0.64–0.70 (bf16) and 0.65–0.85 (fp32) of the
  v0.5.0 kernel time — 1.2–1.6× faster at the median, 2.5× at best
  (32×16384 fp16), with the gain growing with row width. At 32×768 and
  32×1024 the ratios are 0.834–1.023 across ops and dtypes: little to gain,
  and the single row above 1 (`rms_norm` bf16 32×768, 1.84 → 1.88 µs) is
  inside the control group's ±4 %, a tie rather than a demonstrated
  regression. For fp16/bf16 the per-op medians (−30 % to −38 %) are at or
  beyond the PR's H200 medians (−24 % to −36 %); fp32's (−15 % to −35 %)
  lie inside the PR's −15 % to −50 %.
* **The warp-per-row fp8 kernel is faster at hidden size 1024**: 0.67–0.70
  at 8192×1024 for `rms_norm_fp8_dynamic`, 0.89–0.90 at 512×1024, and
  0.91–0.95 for the fused-add variant at 8192×1024 — six rows, fp16/bf16
  only. Nothing here covers N = 128–512 or fp32 for this path.
* **No collateral regression**: the 70 rows that run the same kernel in both
  builds sit at a 1.000 median, 0.961–1.045, which is also the noise floor
  for every other number in this directory. Exactly four rows exceed 1.02:
  three are control rows (`rms_norm` fp16 256×4096 1.045,
  `fused_add_rms_norm_fp8_dynamic` fp32 4096×4096 1.038,
  `fused_add_rms_norm` fp16 4096×4096 1.027) and the fourth is the 32×768
  bf16 row above.
* Graph-replay ratios run 0.03–0.06 closer to 1 than kernel ratios (the
  graph's fixed per-launch cost is the same in both builds) and never change
  the direction of a result. Within a run, the graph min–max spread is 1.8 %
  at the median over all 432 library rows; 25 rows spread more than 5 %, the
  worst 36 % (`fused_add_rms_norm_fp8_dynamic` fp16 192×4096, branch).
* Single run per shape and dtype. No Nsight Compute on this machine, so the
  PR's instruction-count explanation is not re-verified here. Nothing in this
  directory tests the PR's bitwise claims (those rest on the PR's own H200
  and A100 checks). The competitor candidates (aten, eager composite,
  `torch.compile`) are in the files but were not analysed for this release:
  the `torch.compile` candidate ran under the raised recompile limit, and the
  H100 story for the competitors is the one in
  `../2026-08-25_h100-80gb_v050_ops/`.
* The fp32 accuracy question the few-row kernel raises (summation order)
  was answered on the same machine with a separate 16,800-case sweep; the
  rule that holds and its limits are stated in `CHANGELOG.md` under 0.6.0.
  That sweep's raw output is not committed.
