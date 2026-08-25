# Changelog

## 0.5.0 — 2026‑08‑25 — fast backward, GELU backward, LayerNorm-family fp8

### Added

* **Vectorised backward kernels.** The dx kernel and the parameter-partials
  kernel gain 16-byte `Vec` variants mirroring the forward's pattern (the
  scalar kernels remain the odd-N/misaligned fallback and the fp64 gradcheck
  path); the partials block shrinks from 1024 threads/5-level reduce to
  (32,8)/3-level with per-thread register accumulators; the chunk policy now
  also fills the GPU at small M; and one fused finalize kernel replaces
  `partials.sum(0).to(dtype)` ×2 (up to 7 device steps → 3 launches). The
  fwd-train ops also stop materialising zero cotangents for the stats
  outputs where the ctx supports it. Determinism contract unchanged: no
  atomics, fixed chunk grid, fixed reduction orders — dx/dgamma/dbeta remain
  bitwise run-to-run reproducible (tested, including the new chunk-policy
  path). **Cross-version note:** dgamma/dbeta bit values differ from v0.4.2
  (different reduce split and stage-2 order).
* **`layer_norm_gelu` backward** (erf and tanh) — the op table's last
  "fallback" cell is gone. Implemented as a `DGrad` cotangent-transform hook
  on the backward kernels: `dh = dy * gelu'(h)` with `h = xhat*gamma + beta`
  recomputed in-kernel (no extra M×N tensor saved); the fwd-train launcher
  gains the GELU epilogues, and grad-requiring eligible calls route to the
  new `layer_norm_gelu_fwd_train` custom op. Gradcheck-verified (fp64, both
  approximations). **Behavior change** (as v0.4.0's grad-mode change was):
  grad-requiring `layer_norm_gelu` calls no longer fall back to eager.
* **LayerNorm-family fp8 outputs**: `layer_norm_fp8` and
  `fused_add_layer_norm_fp8` (static + dynamic scales, bias included) — the
  LN mirror of the RMS fp8 path, sharing the same epilogues, byte contract
  and NaN-propagating dynamic scale.
* Suite grows 260 → 289 tests; the fuzz gains backward (RMS) and plain-LN
  fp8 configs.

### Measured (2026‑08‑25, A100‑SXM4‑40GB, locked clocks, clean clone `b38a935`)

* **The training step is no longer the weak spot at production shapes.** A
  full fwd+bwd step vs PyTorch autograd (kernel time): fp16 LN 0.75–1.42×,
  RMS 0.82–1.44×, LN+GELU 0.80–1.44× — ≥ 1.16× at every M ≥ 2048 shape —
  and fp32 LN 0.83–1.13×, RMS 0.86–1.16×, LN+GELU 0.95–1.88×. v0.4.2
  measured 0.43–1.10×. Still published: 512×1024 stays 0.75–0.95× (both
  sides launch-bound, ours pays more launches) and small-M wall clock stays
  dispatch-dominated (~0.5×), amortised by torch.compile/CUDA graphs.
* New fp8 ops (fp16, kernel time): `layer_norm_fp8` 3.5–6.0× vs the eager
  chain and 1.02–1.49× vs the compiled chain at M ≥ 2048 — but **0.75× at
  512×1024**: the "≥ 1× at every shape" fp8 claim remains deliberately
  RMS-only (re-measured this release: 1.01–1.76× fp16, 1.10–1.69× fp32).
  `fused_add_layer_norm_fp8`: 3.6–5.7× / 1.25–1.68× (0.74× at 512×1024).
* Forward rows for pre-existing ops re-validate v0.4.2 within noise; full
  tables in `benchmarks/results/2026-08-25_a100-40gb_v050_ops/`.

### Measured — H100 addendum (2026‑08‑25, H100 80GB HBM3, locked 1830 MHz)

* First non-A100 data
  (`benchmarks/results/2026-08-25_h100-80gb_v050_ops/`, clean clone of the
  v0.5.0 data commit `8af8dbb` built for sm90; 289 tests × 3 kernel modes + fuzz green — correct on
  Hopper, fp8 converts now native). The eager-mode wins transfer and grow
  (`rms_norm_fp8` 7.7–11.4× vs the eager chain in fp16; fused-add 1.0–1.7×;
  training up to 1.79× of autograd; 87 % of datasheet peak), but **the
  kernel-time edge over `torch.compile` does not transfer** — Inductor's
  Hopper codegen beats these A100-tuned launch heuristics at many shapes
  (RMS fp8: 0.69–1.20× vs compiled) — so every "≥ 1× vs compiled" claim is
  now explicitly scoped to the A100. Wall clock: the RMS fp8 ops still beat
  the compiled chain at every fp16 shape (1.00–3.40×). Clock note: the
  H100 pins at its *sustained* 1830 MHz rather than the 1980 boost (which
  it cannot hold under load; methodology §7 applies unchanged).

## 0.4.2 — 2026‑08‑25 — dynamic-fp8 NaN-scale fix

Found by a randomized contract fuzz (130 random shape/dtype configs) run as a
post-release verification pass: v0.4.1 fixed NaN handling in the fp8 *value*
path but not in the dynamic *scale* path. The fuzz is now committed as
`tests/fuzz_contracts.py` (run explicitly; not part of the pytest suite).

* **Dynamic fp8 scale on a NaN row was silently finite.** The per-row amax
  reduction used `fmaxf` (and a plain `>` select in the warp shuffle), both of
  which DROP a NaN operand — so a NaN-poisoned row, whose normalised values
  and quantised bytes are all NaN, still got the 1e-12 amax floor and a scale
  of ~2.2e-15 instead of NaN. The eager composite (`torch.amax`) propagates
  NaN, so the kernel and its documented "numerically identical" fallback
  disagreed. The amax accumulation now runs as an integer max over the float
  bit patterns of `|y|` (for non-negative floats, IEEE ordering equals integer
  ordering and every NaN pattern compares above +inf — branch-free NaN
  propagation at no extra memory traffic), the warp reduction and the
  `scale_ub`/floor clamps propagate NaN explicitly, and a regression test
  pins values, scale, and the clean-row bytes around a poisoned row in all
  three dtypes.
* Measured cost of the correct semantics: ~2–4 % kernel time on the two
  fp8-dynamic ops only (the amax pass gains one integer-max per element);
  every other op is untouched.

### Measured (2026‑08‑25, A100‑SXM4‑40GB, locked clocks)

* This release's data
  (`benchmarks/results/2026-08-25_a100-40gb_v042_ops/`, clean clone of
  `c24b55a`, clock state recorded in the JSON metadata) re-measures
  everything. The fp8 headline survives its own fix:
  ≥ 1× vs the `torch.compile`'d chain at every shape — now 1.03–1.73×
  (`rms_norm_fp8`) and 1.01–1.74× (`fused_add_rms_norm_fp8`) in fp16, with
  the 512×1024 margin thinned to 1.01–1.03× by the NaN fix. Full tables in
  the results README.
* **Benchmark methodology change**: committed runs now lock SM clocks at the
  boost ceiling (`nvidia-smi -lgc`) first. Discovered re-measuring v0.4.1:
  with default unlocked clocks, ~10 µs kernels measured up to 29 % slower
  than on release night (1095 MHz application clocks vs 1410 MHz boost —
  short profiler loops don't reliably ramp), while ≥ 100 µs kernels
  reproduced within ~2 %. Locked clocks reproduce the committed v0.4.1
  numbers, and ratios at small shapes are now stable day-to-day.
  `docs/methodology.md` §7 documents it. Consequence: v0.4.2 numbers are
  not directly comparable to the v0.4.1 directory at the smallest shapes
  (both directories' READMEs say so).

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

Re-measured from a clean clone of this release's code commit `1193a53`
(`git_dirty=false`), committed under
`benchmarks/results/2026-08-24_a100-40gb_v041_ops/` — the v0.4.0 directory
stays as the historical record. The fairness fix moves the fp8-vs-eager
ratios from 5.6–8.6× to the honest 5.2–7.2×, and the fp8 ops now beat the
`torch.compile`d chain at **every** measured shape (1.06–1.79×). Everything
else re-validates within noise: fused_add_rms_norm fp16 1.23–1.58× vs eager
(0.94–0.99× vs compiled at M ≥ 2048), rms_norm 0.99–1.27× vs aten, training
step 0.44–1.09× fp16 / 0.54–1.28× fp32.

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
  parity (1.00×) at 8192 × 1024, 0.93× at 512 × 1024. The results directory's README has the
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
