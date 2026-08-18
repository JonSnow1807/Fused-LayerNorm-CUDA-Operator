# Historical (July 2025) results for a kernel that no longer exists

**Nothing in this folder applies to the current kernel, and no speedup number
from it is cited as evidence anywhere in the repository.** The only use made
of these files is the PyTorch-side large-row timings quoted, with caveats, in
[`../../../docs/methodology.md`](../../../docs/methodology.md) §6. The files are kept, unmodified,
purely for transparency about what the project has published in the past.

## What they describe

These files were committed on 2025-07-17 (commit `2924f53`, "Final changes")
together with the *original* kernel: a forward + backward implementation with
float4 vectorised loads and shared-memory / warp-shuffle reductions
(`git show 2924f53:csrc/layernorm_cuda_kernel.cu`, 368 lines in that commit's
diffstat, plus `csrc/layernorm_cuda_kernel_optimized.cu`).
`csrc/layernorm_cuda_kernel.cu` was **replaced** by the simple forward-only
kernel in commit `3686b4c` on 2025-08-17 ("feat: Achieve 2.5x speedup by
simplifying kernel", 131 insertions, 291 deletions in that one file);
`csrc/layernorm_cuda_kernel_optimized.cu` was left in the tree but no longer
built (commit `4ea14d2`'s `setup.py` lists only `bindings.cpp` and
`layernorm_cuda_kernel.cu`, not the `_optimized` file)
and is removed in this rewrite. The current kernel is a further rewrite of the
`3686b4c` one. Nothing measured here was ever run against a kernel that exists
in the tree today.

## Provenance problems

* **No script in the repository history writes any of these files.** The July
  benchmark script (`git show 2924f53:benchmarks/benchmark_layernorm.py`)
  writes `benchmark_results.csv` / `benchmark_results.json` /
  `benchmark_summary.json`; the plotting script
  (`benchmarks/visualize_results.py` at the same commit) writes
  `speedup_by_model.png`, `speedup_vs_batch_size.png`, `speedup_heatmap.png`,
  `dtype_comparison.png` and `performance_table.md`. None of the file names in
  this folder (`achievement_results.json`, `ACHIEVEMENT.md`,
  `final_benchmark_results.csv`, `large_model_results.csv`,
  `large_model_summary.json`, `portfolio_summary.json`,
  `portfolio_performance.png`, `portfolio_table.png`) is produced by any
  committed script (checked with `git grep` across every commit).
* **`achievement_results.json` was not produced by any committed script.** Its `"timestamp"` field is
  the literal, unexpanded string `"$(date -Iseconds)"` — a shell substitution
  that never ran — so whatever wrote the file did not evaluate that
  expression; no benchmark in the history emits this file.
* **The same-day files contradict each other.**
  * `large_model_results.csv` / `large_model_summary.json` (timestamp
    2025-07-17T02:54:23): six configs at 4096 rows (batch x seq = 8x512,
    4x1024, 2x2048) with hidden sizes 1600–12288, PyTorch 0.056–0.375 ms vs
    fused 0.08–0.397 ms, **speedups 0.687–1.027x** (`best_speedup` 1.027,
    `average_speedup_large_models` 0.9803) — i.e. the July kernel was mostly
    *slower* than PyTorch at large row counts.
  * `final_benchmark_results.csv`, `portfolio_summary.json` (timestamp
    2025-07-17T03:18:03), `portfolio_performance.png` and `portfolio_table.png`
    (about 24 minutes — 23 min 40 s — later): six configs at 1024–2048 rows with hidden sizes
    4096–8192, PyTorch 0.069–0.102 ms vs fused 0.049–0.069 ms, **speedups
    1.416–1.488x**, all flagged `meets_target = True` against a "1.4x target";
    `achievement_results.json` / `ACHIEVEMENT.md` repeat "1.434x" (2048 x 4096)
    and "1.461x" (2048 x 5120), which match neither CSV exactly.
  * The 4096-row rows of `large_model_results.csv` (hidden 4096: 0.969x;
    hidden 5120: 1.027x) and the 2048-row rows of `final_benchmark_results.csv`
    (hidden 4096: 1.416–1.425x; hidden 5120: 1.488x) were produced within half
    an hour of each other on the same GPU (both say NVIDIA A100-SXM4-80GB) and
    are not reconciled anywhere.
* All files record only milliseconds to 2–3 decimals and a ratio; no
  methodology, iteration counts, warm-up, dtype, or torch/CUDA version is
  recorded in any of them.

## What is worth keeping from them

Only one thing: `large_model_results.csv` is the only place in the repository
with any timing at **thousands of rows**, and there PyTorch's own LayerNorm
took 0.056–0.375 ms for 4096 rows x 1600–12288 columns (dtype not recorded;
if fp32, that is roughly 0.9–1.2 TB/s of read+write traffic). That is the
regime where LayerNorm is memory-bound; see
[`../../../docs/methodology.md`](../../../docs/methodology.md) for what that
implies for a block-per-row kernel. It is *not* a measurement of the current
kernel.

## Files

| file | content |
|---|---|
| `achievement_results.json` | summary not produced by any committed script; literal `$(date -Iseconds)` timestamp; "1.434x" and "1.461x" |
| `ACHIEVEMENT.md` | prose version of the same claim ("Successfully Achieved 1.4x+ Speedup") |
| `final_benchmark_results.csv` | 6 rows, 1024–2048 rows x 4096–8192, speedups 1.416–1.488x |
| `large_model_results.csv` | 6 rows, 4096 rows x 1600–12288, speedups 0.687–1.027x |
| `large_model_summary.json` | JSON of the previous CSV plus `best_speedup` 1.027, average 0.9803, timestamp 2025-07-17T02:54:23 |
| `portfolio_summary.json` | JSON of `final_benchmark_results.csv` (rounded to 2 decimals), `best_speedup` 1.488, timestamp 2025-07-17T03:18:03 |
| `portfolio_performance.png` | bar chart of the six `final_benchmark_results.csv` speedups with a dashed "Target (1.4x)" line |
| `portfolio_table.png` | rendered table of the same six rows |

Do not delete these files; they are the record. Do not cite their speedups as evidence for anything.
