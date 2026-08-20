# Changelog

## 0.3.0 — 2026‑08‑20 — first hardware run + vectorised kernel

The 0.2.0 tree, written on a machine without CUDA, was built and run for the first time on an
NVIDIA A100‑SXM4‑40GB (PyTorch 2.13.0+cu129, CUDA toolkit 12.9).

* **Verified:** all 102 tests pass unmodified; CUDA‑graph capture works (the 0.2.0 caveat about it
  being untested is retired); `bench_layernorm.py` ran without changes and its fp32/fp16 output is
  committed under `benchmarks/results/2026-08-20_a100-40gb_kernel_time/`.
* **Measured, then fixed:** the scalar two‑pass kernel was 1.3–2.0× faster than PyTorch's kernel at
  small/latency‑bound shapes but 0.42–0.98× at large memory‑bound ones — exactly the hypothesis
  recorded in the 0.2.0 README (three passes over the row, scalar loads). Added a second kernel:
  16‑byte vectorised loads (4 × fp32 / 8 × fp16‑bf16 / 2 × fp64) + single‑pass Welford statistics
  with Chan's parallel merge — the same algorithm as PyTorch's `vectorized_layer_norm_kernel` — and
  a measured selection heuristic (vectorised iff N divisible by the vector width, N ≥ 128, M ≥ 256;
  block size ≈ two vectors per thread in [64, 256]). Result (fp32 kernel time): faster than PyTorch
  on 8 of 11 benchmark shapes (1.02–1.97×), 0.88–0.98× on the remaining three (many narrow rows).
  The README's measurement section has the full table.
* fp16 initially used 4‑element (8‑byte) loads and measured ~0.8× of PyTorch at large shapes;
  switching to 16‑byte loads (8 halves) brought it to parity or ahead except at three shapes
  (0.69×/0.89×/0.97×), which the README lists.
* Version 0.2.0 → 0.3.0 (`pyproject.toml`, `setup.py`, bindings, package, test).
* No API changes; the scalar kernel and every 0.2.0 behaviour (dtype rules, fallbacks, forward‑only
  semantics) are unchanged.

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
