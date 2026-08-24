# Changelog

## 0.4.1 — 2026‑08‑24 — post-release integrity audit fixes

An adversarial audit of the pushed v0.4.0 (claims-vs-data, kernel review,
Python review, hygiene, benchmark-methodology review) plus the first real CI
runs produced this patch release. In the spirit of this repository's ledger,
the defects are listed rather than paraphrased away.

### The v0.4.0 tag itself was defective

* `import fused_layernorm` **failed on torch 2.4** — the declared minimum —
  because `torch.library`'s schema inference there cannot resolve the string
  annotations produced by `from __future__ import annotations` (and, once
  fixed, also rejects string *default values*: `approximate` is now
  `Optional[str] = None`, mapped to `"none"` in the op body). Both were
  caught by the torch-2.4 CI leg on the first real Actions run.
* The CUDA CI job referenced a nonexistent container image
  (`pytorch/pytorch:2.13.0-cuda12.9-cudnn9-devel`); it now uses the real
  `2.13.0-cuda12.6-cudnn9-devel` tag. CI is green (both torch legs + the
  sm80/sm90 compile job) as of this release.
* The stale `v2.0.0` tag — which served the pre-rewrite README with the
  claims the 0.2.0 audit retracted — has been deleted from the remote.

### Claims corrected against the release's own data

* "On wall clock they beat both everywhere measured" was **false**: the
  compiled composite wins fp32's largest shape by ~15 % (0.85–0.87× at
  4096×8192) and ties fp16's two largest. README/CHANGELOG now state exactly
  that.
* The >100 %-of-peak bandwidth rows were explained by a wrong mechanism (a
  "re-read" the bytes models do not contain); the real cause is inter-call
  L2 residency of 8–34 MB working sets across the 200 timed calls.
* Range fixes: the "4–20×" and "4–9×" wall-clock advantages are the measured
  4.5–11.2×; the fused-add kernel/bandwidth ranges are stated per op instead
  of blurring the RMS and LayerNorm variants together.

### Code fixes

* Removed `__restrict__` from the fused-add kernels' residual pointer pair —
  the in-place variant aliases them, and promising no-alias there was UB
  (benign under current codegen, still wrong).
* fp8: NaN inputs now quantise to NaN (fmin/fmax silently dropped the NaN
  and produced −448); `scale_ub` must be positive (the kernel treats ≤ 0 as
  "no clamp", the eager fallback clamped to zero — the divergence is now
  rejected at the API).
* Empty-shape semantics: zero-size inputs return zeroed statistics/scales
  instead of uninitialised memory, and an empty-batch backward returns
  zeros(N) parameter grads instead of "no grad".
* `_eligible` gained the kernel dtype whitelist, so unsupported dtypes
  (complex, integer) take the PyTorch fallback instead of the extension's
  error path; the fp8 grad-guard now also covers a grad-requiring `scale`.
* Benchmark fairness: the eager fp8 composite no longer computes `y.float()`
  twice (that handicapped the competitor — fp8 claims are re-measured in
  this release's data), the fp8 correctness gate checks the scale
  independently, and the backward rows are labelled as full fwd+bwd
  training-step numbers, which is what they time.
* A sweep of stale pre-v0.4.0 prose (forward-only claims in module
  docstrings, bindings header, examples, methodology notes, benchmark READMEs,
  test names, pyproject description/keywords).

### Measured

Re-measured from a clean clone of this release's code commit (the fairness
fix above changes the fp8 competitor): see
`benchmarks/results/` — the v0.4.1 directory README carries the updated
numbers and provenance. (Filled by the release data commit.)

## 0.4.0 — 2026‑08‑24 — from one kernel to a fused-normalisation library

### Added

* **RMSNorm**: `rms_norm()` (exact `F.rms_norm` drop-in, including the
  `eps=None` = machine-epsilon-of-compute-dtype semantics), `RMSNorm`
  (`nn.RMSNorm` subclass, `from_torch` parameter sharing), `replace_rmsnorm`.
* **Fused residual-add + norm** — the op eager `aten` does not have:
  `fused_add_layer_norm` / `fused_add_rms_norm` returning
  `(out, new_residual)`, with the bitwise contract
  `new_residual == round(x + residual)` (rounded exactly once) and
  `out == plain_norm(new_residual)` (statistics over the rounded sum;
  asserted with `torch.equal` across dtypes and kernel flavours).
  `inplace=True` (inference-only, contiguous residual required, rejected
  under grad) mutates the residual in place. `FusedAddLayerNorm` /
  `FusedAddRMSNorm` modules return `(normed, new_residual)` and downgrade
  inplace to out-of-place under grad mode.
* **Real CUDA backward** for `layer_norm`, `rms_norm` and both fused-add ops:
  dx via a paired two-sum row reduction; dgamma/dbeta via a deterministic
  two-stage reduction (fixed-chunk fp32 partials + fixed-shape aten sum — no
  atomics, bitwise run-to-run reproducible, tested). Fused-add backward
  accepts cotangents for BOTH outputs (`dx = dresidual = norm_dx + dz`).
  Verified with `torch.autograd.gradcheck` on CUDA fp64 (the reason fp64 is
  supported on these paths), composite-grad comparisons for fp32/16/bf16, and
  a different-cotangents both-outputs test.
* **fp8-E4M3 quantised outputs** (inference-only, RMS family):
  `rms_norm_fp8` / `fused_add_rms_norm_fp8` with static per-tensor scale
  (a `[1]` fp32 CUDA tensor read on-device — no host sync, CUDA-graph
  capturable) or dynamic per-token scales (in-kernel row amax, optional
  `scale_ub`, all-zero-row guard, trailing broadcast dim). Byte-level
  contract: output equals quantising the plain norm output with
  `round_e4m3(clamp(y * (1/scale), ±448))` — reciprocal multiply, rounded
  through the input dtype first (composite equivalence).
* **`torch.compile` integration**: every public op is a
  `torch.library.custom_op` (namespace `fused_layernorm::`) with fake impls
  and, for the training forwards, `register_autograd` — the wrappers trace
  `fullgraph=True` with zero graph breaks (pinned by tests via
  `torch._dynamo.explain`), including compiled backward and functionalised
  inplace ops. Eager no-grad calls take a raw-pybind fast path
  (`torch.compiler.is_compiling()` split) because the custom-op dispatcher
  measured ~40–50 µs per call.
* **CI** (`.github/workflows/ci.yml`) — the job promised since 0.2.0: a CPU
  test matrix (py3.10/torch 2.4, py3.12/torch 2.13) and a CUDA compile-only
  job (sm80/sm90). CI has no GPU and says so; hardware truth stays with the
  committed benchmark provenance files.
* `benchmarks/bench_norms.py`: op-registry benchmark for the new op family on
  the verified timing core, with `torch.compile`d composites as competitors
  and per-op bytes-moved models recorded in the JSON.
* csrc groundwork: shared device machinery extracted verbatim into
  `norm_{reduce,vec,epilogue,dispatch}.cuh`; generic forward template
  (`norm_fwd_kernels.cuh`) covering {LayerNorm, RMSNorm} × {plain, fused-add}
  × epilogue functors in scalar and 16-byte-vectorised flavours; the two
  verified v0.3.0 LayerNorm kernels are byte-identical and still serve plain
  LayerNorm inference.

### Changed

* **Gradient-requiring calls now run the fused kernels** (fwd-train variants
  whose outputs are bitwise identical to the inference path) instead of
  falling back to PyTorch — the one intended behaviour change; the old
  fallback tests were updated to pin the new routing.
* torch floor 2.0 → 2.4 (`torch.library.custom_op`).
* Version single-sourced from `fused_layernorm/__init__.py` (setup.py injects
  it into the extension; pyproject reads it dynamically; tests assert
  equality, not literals).
* Test suite: 104 → 257 tests.

### Found the hard way (documented in code)

* `__nv_fp8_e4m3(unsigned char)` numerically converts the storage byte — the
  raw byte must be assigned to `.__x` (caught by the byte-equality tests).
* Reciprocal-multiply vs division round differently — the quantisation
  contract names the multiply.
* `ctx.needs_input_grad`'s length varies across custom-op variants (indexed
  defensively) and custom-op outputs may not alias each other (distinct
  empty tensors for unrequested grads).
* Routing eager calls through the custom-op dispatcher costs ~40–50 µs per
  call — measured by the benchmark this release adds, fixed with the
  `is_compiling()` split.

### Measured

Committed under `benchmarks/results/2026-08-24_a100-40gb_v040_ops/` (fp16 and
fp32, produced from a clean clone of release code commit `5fdb217`,
`git_dirty=false`; 257 tests passed in the same clone first). Kernel-time
ratios over six shapes (512×1024 … 4096×8192), fp16: `fused_add_rms_norm`
1.22–1.58× vs the eager composite (0.95–0.99× vs the `torch.compile`d
composite at M ≥ 2048; on wall clock faster than the eager composite
everywhere and than the compiled one at all but the two largest fp16 shapes,
which tie — in fp32 the compiled composite wins the largest shape by ~15 %);
`fused_add_layer_norm` 1.00–1.49× vs eager; `rms_norm` 0.99–1.27× vs aten's
fused `F.rms_norm` kernel; the fp8 ops 5.6–8.6× vs the eager
norm→amax→cast chain and **1.44–1.81× vs the compiled chain at M ≥ 2048** —
the one outright win over Inductor. Deliberately published weak spots: at
512×1024 Inductor's kernels beat ours on pure kernel time, and a full
training step (fwd+bwd) measures 0.44–1.10× of PyTorch's autograd — the
backward is correctness-first; speed there is future work. fp32 ranges are in
the same directory's README.

## 0.3.0 — 2026‑08‑20 — first hardware run + vectorised kernel

The 0.2.0 tree, written on a machine without CUDA, was built and run for the first time on an
NVIDIA A100‑SXM4‑40GB (PyTorch 2.13.0+cu129, CUDA toolkit 12.9).

* **Verified:** all 102 tests of the 0.2.0 suite pass unmodified (104 after the regression tests
  added below); CUDA‑graph capture works (the 0.2.0 caveat about it being untested is retired);
  `bench_layernorm.py` ran without changes and its fp32/fp16 output — plus a scalar‑baseline run —
  is committed under `benchmarks/results/2026-08-20_a100-40gb_kernel_time/`.
* **Measured, then fixed:** the scalar two‑pass kernel is 1.3–2.0× faster than PyTorch's kernel at
  small/latency‑bound shapes but 0.42–0.98× at every M ≥ 512 shape (committed
  `FUSED_LAYERNORM_FORCE_KERNEL=scalar` baseline file) — exactly the hypothesis recorded in the
  0.2.0 README (three passes over the row, scalar loads). Added a second kernel: 16‑byte vectorised
  loads (4 × fp32 / 8 × fp16‑bf16 / 2 × fp64) + single‑pass Welford statistics with Chan's parallel
  merge — the same algorithm as PyTorch's `vectorized_layer_norm_kernel` — and a measured selection
  heuristic (vectorised iff N divisible by the vector width, N ≥ 128, M ≥ 256, all pointers
  16‑byte aligned; block size ≈ two vectors per thread in [64, 256]). Result (fp32 kernel time,
  committed clean‑provenance run): faster than PyTorch on 9 of 11 benchmark shapes (1.01–1.97×),
  parity (1.00×) at 8192 × 1024, 0.93× at 512 × 1024. The README's measurement section has the
  full table.
* fp16 initially used 4‑element (8‑byte) loads and measured well behind PyTorch at large shapes
  (interim development run, data not retained); switching to 16‑byte loads (8 halves) — and, in
  the hardening pass below, loading `weight`/`bias` as whole vectors — brought it to faster
  everywhere (1.02–1.98×) except 512 × 1024 (0.74×).
* Version 0.2.0 → 0.3.0 (`pyproject.toml`, `setup.py`, bindings, package, test).
* No API changes; the scalar kernel and every 0.2.0 behaviour (dtype rules, fallbacks, forward‑only
  semantics) are unchanged.
* Still missing: the CI workflow (compile‑only nvcc job + CPU‑only pytest job) that the 0.2.0 notes
  promised in a follow‑up commit was never built; the promise is repeated here rather than silently
  dropped.
* Post‑release hardening after an adversarial review of this release: the vectorised path now
  checks 16‑byte *pointer* alignment at launch instead of assuming it from contiguity — a
  contiguous 1‑D slice (`base[1:]`) keeps its storage offset, and such an input (or weight/bias)
  previously crashed the vectorised kernel with "misaligned address"; a regression test covers it.
  Added `FUSED_LAYERNORM_FORCE_KERNEL=scalar|vec` (debug/benchmark override), driver/env fields in
  the benchmark metadata, and a scalar‑baseline benchmark run so every number this repository
  quotes about the scalar kernel points at a committed file.

## 0.2.0 — 2026‑08‑18 — honesty rewrite

This release is a correction, not a feature release. An audit of the repository as of commit
`12dee09` (2025‑08‑17) checked every claim in the README against the code, the committed data files
and the git history. The findings, and what was done about each, are listed here so that the record is
explicit.

### Claims removed from the README (not supported by any committed artifact)

| Former claim | What the repository actually contains | Action |
|---|---|---|
| "1.86×–2.66× faster than PyTorch" and the per‑configuration table (BERT 2.36×/2.57×, GPT‑2 1.73×/2.61×, GPT‑3 1.64×/2.64×, Large 1.74×/2.67×, Odd 1.81×/2.83×) | No committed file contains that per‑configuration table or the 1.86×/2.66× averages. `publication_results.json` (same five shapes, produced by `tests/publication_ready_benchmark.py`, whose logic matches `reproduce_speedup.py`) gives 2.50/1.59/1.68/1.73/1.78× and 2.51/2.55/2.49/2.52/2.76× (averages 1.85×/2.57×); `publication_validation_results.json` gives averages 1.83×/2.13×. (1.73 and 2.57 do occur in the first file, but for a different row and as the cached average.) The README's averages are exactly the means of its own table. | Removed. Committed numbers are now shown as absolute µs per call with the methodology stated. |
| "Memory bandwidth utilisation 10.7 %", "actual bandwidth 167 GB/s", "LayerNorm is latency‑bound, uses ~10 % of bandwidth" | The only committed bandwidth figure is 26.4 GB/s / 1.70 % (one event‑timed call at 32 × 4096 with no warm‑up loop of its own, `publication_validation_results.json → bandwidth`). 167 GB/s and 10.7 % are consistent with each other (167 / 1555 = 10.7 %); neither appears in any committed file, and no script in the history computes them. | Removed. |
| "A100 peak 1555 GB/s" | Every committed file that records a GPU names an **A100‑SXM4‑80GB** (`publication_results.json` records none), whose datasheet peak is 2 039 GB/s; 1 555 GB/s is the 40 GB part. `tests/publication_validation.py` hard‑coded 1555 for any device containing "A100". | Corrected in README/docs; the new benchmark uses a per‑device table. |
| "Removing vectorization made it faster" / "float4 breaking on odd dimensions" | The kernel replaced in commit `3686b4c` (`git show 3686b4c^:csrc/layernorm_cuda_kernel.cu`) contained **no** float4 loads (only an unused `VectorizedLoader` typedef mentioning `float4`); the only float4 load/store code (`layernorm_cuda_kernel_optimized.cu`) was dispatched only for N ≥ 4096 and had a scalar fallback for N % 4 ≠ 0. The pre‑`3686b4c` scalar kernel launched `min(N, 1024)` threads and its block reduction dropped partial warps, so any N not a multiple of 32 was mis‑reduced; whether that is the "odd dimensions" bug the old README refers to is not recorded. No before/after A/B under one harness exists. | Removed. |
| "Better numerical accuracy — 4.77e‑07 maximum error" | 4.77 × 10⁻⁷ is the max absolute **difference from PyTorch** at 32 × 4096 fp32 with weight = 1, bias = 0. Nothing compares either implementation to a higher‑precision reference. For fp64 inputs the old kernel accumulated in fp32 (expected error of order 1e‑2 on rows like `1000 + 1e-3·randn` — an estimate, not a measurement; see `tests/test_layernorm.py::test_kernel_float64_ill_conditioned_rows`). | Reworded; fp64 now accumulates in double. |
| "~100 lines vs PyTorch ~500 lines", "register pressure high/low", "instruction complexity high/low" | Old kernel file: 208 lines. PyTorch v2.7.1 `layer_norm_kernel.cu`: 1 502 lines (forward ≈ 460, incl. fp16/bf16/fp64, autograd outputs, ROCm). No ptxas/ncu data exists for either. | Removed; factual description of PyTorch's kernel added. |
| "Statistical significance p < 0.0001, 30 samples, 12+ configurations, edge cases 1×1 to 200×10000" | Committed statistics cover 3 configurations; the 12‑configuration script (`statistical_validation.py`) would fail at `json.dump` (`numpy.bool_` is not JSON‑serialisable) and no `statistical_results.json` was ever committed; its threshold was 0.001; no committed result covers 200 × 10000. The t‑tests compared two near‑constant host overheads. | Removed. |
| "Production ready", "universal compatibility — works on ANY dimension", "mixed precision (FP16/FP32) support", "25 % memory reduction", "gradient checkpointing compatible" (package docstring) | The built extension had no input validation, launched on the legacy default stream, had no device guard or launch check, dispatched only fp32/fp64, accepted only 2‑D contiguous input, had no backward, and the Python package called extension functions (`forward`, `backward`, `get_memory_usage`) that did not exist. | Removed; see "Code changes". |
| Usage snippets (`layernorm(x, gamma, beta, eps=1e-5)`; `setattr(module, name, lambda ...)`) | Both raise `TypeError`: bindings had no `py::arg` names, and `nn.Module.__setattr__` refuses a function in place of a registered submodule. | Replaced with working examples; bindings now have keyword arguments. |
| `tests/test_correctness.py`, `tests/numerical_validation.py` | Never existed in any commit. | References removed; real test suite added. |
| "Realistic (different tensors) vs Optimal (cached tensor reuse)" | The two modes differ in timing method (one call between events on an idle GPU vs 500 or 1000 pipelined calls after warm‑up, depending on the script), not in cache state; the input was generated on‑GPU immediately before timing in both. | Explained in `benchmarks/legacy/README.md` and `docs/methodology.md`. |

### Files removed or quarantined

* Deleted (dead or targeting an API that no longer exists): `csrc/layernorm_cuda.cpp`,
  `csrc/layernorm_cuda_kernel_optimized.cu`, old `csrc/layernorm.h`, `fused_layernorm/functional.py`,
  old `fused_layernorm/layernorm.py`, old `tests/test_layernorm.py`, `benchmarks/benchmark_layernorm.py`,
  `benchmarks/visualize_results.py`, the empty `docs/architecture.md` and `docs/optimization_guide.md`,
  the empty old `examples/example_usage.py` (replaced by a real example), four tracked `.DS_Store` files.
* Deleted (scripts that produced no committed artifact and whose checks reused the single‑call
  event‑pair timing described above; `statistical_validation.py` would additionally fail at
  `json.dump`): `tests/break_kernel_test.py`, `tests/prove_not_hardcoded.py`,
  `tests/stable_measurement.py`, `tests/statistical_validation.py`, `tests/edge_cases.py`.
* Moved unmodified to `benchmarks/legacy/`: `publication_validation.py` and
  `publication_ready_benchmark.py` (which produced the committed August data) and
  `reproduce_speedup.py` (the script the old README told users to run; writes no file).
* Moved to `benchmarks/results/2025-08-17_a100_eager_latency/`: `publication_results.json`,
  `publication_validation_results.json` (the only committed data; taken with the `12dee09`
  predecessor of the current kernel), with a README.
* Moved to `benchmarks/results/historical_2025-07_deleted_kernel/`: the July‑2025 "1.4× achieved"
  files (`ACHIEVEMENT.md`, `achievement_results.json` — which contains a literal, unexpanded
  `$(date -Iseconds)` timestamp —, `final_benchmark_results.csv`, `large_model_results.csv`,
  `large_model_summary.json`, `portfolio_*.{png,json}`). They describe a forward+backward kernel deleted
  in `3686b4c`, were not produced by any script in the history, and disagree with each other
  (0.69–1.03× vs 1.42–1.49× on the same day). Kept only as a record.

### Code changes

* `csrc/layernorm_cuda_kernel.cu`: single templated kernel (`scalar_t`, `acc_t`, GELU flag);
  fp16/bf16 dispatch added; accumulation in `at::acc_type` (double for fp64); launch on
  `at::cuda::getCurrentCUDAStream()` under `c10::cuda::CUDAGuard`; `C10_CUDA_KERNEL_LAUNCH_CHECK()`;
  64‑bit row offsets; power‑of‑two block size in [32, 1024] chosen from N; `weight` and `bias`
  handled independently; erf‑GELU by default (matches `F.gelu`), tanh on request; empty inputs return
  without launching.
* `csrc/bindings.cpp`: `TORCH_CHECK` validation of device/rank/dtype/shape, any‑rank input flattened
  to 2‑D and reshaped back, non‑contiguous inputs made contiguous, `py::arg` names with defaults
  (`weight=None, bias=None, eps=1e-5, approximate="none"`), `__version__`.
* `fused_layernorm/`: rewritten as a thin wrapper — `layer_norm`, `layer_norm_gelu`, `LayerNorm`
  (subclass of `nn.LayerNorm`), `replace_layernorm`, `is_available`; falls back to PyTorch when the
  kernel is unavailable or when autograd is required.
* `tests/test_layernorm.py`: pytest suite with random affine parameters, many shapes and dtypes,
  error cases, streams, GELU forms, empty inputs; skips cleanly without a GPU.
* `benchmarks/bench_layernorm.py`: separates eager latency, profiler kernel time and CUDA‑graph replay;
  records environment; per‑device peak‑bandwidth table.
* `setup.py` / `pyproject.toml` / `requirements.txt`: real metadata, `-O3` only (no fast‑math), minimal
  dependencies. (A CI workflow — compile‑only job in an nvcc container plus a CPU‑only pytest job — is
  prepared in a follow‑up commit; it has not run yet.)

### Known gaps after this release

None of the new code has been executed on a GPU (this rewrite was done on a machine without CUDA).
The GPU test suite and the benchmark need to be run and their outputs committed before any
performance statement is made. See "Not verified / limitations" in the README.

## Pre‑rewrite state (self‑declared 1.0.0) — 2025‑08‑17

State of the repository at commit `12dee09` ("Add comprehensive validation suite"); its `setup.py` and
package declared version 1.0.0. The version was deliberately reset to 0.2.0 for the rewrite above,
because the extension API, the Python package API and every published number changed. Superseded by
the notes above.
