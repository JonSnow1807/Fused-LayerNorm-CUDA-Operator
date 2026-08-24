# 2026-08-24 — A100-SXM4-40GB — v0.4.0 op-family measurements

Both files produced by `benchmarks/bench_norms.py` from a **clean clone of the
v0.4.0 release code commit `5fdb217`** (each JSON's `metadata` records
`git_commit = 5fdb217…`, `git_dirty = false`, `extension_version = 0.4.0`,
driver 570.148.08, `TORCH_CUDA_ARCH_LIST = 8.0`, PyTorch 2.13.0+cu129 / CUDA
12.9). The full 257-test suite passed in the same clone immediately before.
Idle GPU; runs sequential.

* `bench_norms_…_float16.*` — `python benchmarks/bench_norms.py --dtype float16`
* `bench_norms_…_float32.*` — `python benchmarks/bench_norms.py --dtype float32`

Method: per op and shape, a correctness gate first (bitwise for the
contractual properties), then eager per-call latency (median of 20 reps × 200
calls, candidates interleaved), GPU kernel time from `torch.profiler` (the
only number quoted as a kernel speedup), and CUDA-graph replay where
capturable. Competitors: the eager composite, the same composite under
`torch.compile(fullgraph=True)` (warmed; compile time off the clock), and for
plain `rms_norm` aten's own fused kernel via `F.rms_norm`. The
`layer_norm_bwd` / `rms_norm_bwd` rows time a full **forward + backward**
(`torch.autograd.grad`) against the composite's autograd.

Notes for reading the JSONs honestly:

* Each op's `bytes_model` is recorded per row and counts each tensor exactly
  once. A few entries exceed 100 % of the 1 555 GB/s datasheet peak (e.g.
  plain `rms_norm` fp16 at 2048×4096, and compiled candidates at the same
  shapes): at those shapes the working set (8–34 MB) fits the A100's 40 MB
  L2 and stays resident across the 200 timed calls, so much of the modelled
  traffic never reaches DRAM. Ratios between candidates are unaffected
  (same model both sides).
* The wall-clock (`eager_us`) columns are part of the story: the compiled
  composite's kernels are excellent, but each call pays ~90 µs of
  guard/dispatch latency — at 512×1024 fp16 that is 4.5–11.2× more wall
  clock than these ops, depending on the op. The advantage flips at the
  largest fp32 shape, where the compiled composite wins on wall clock too
  (0.85–0.87× at 4096×8192).

Summary (kernel time, ours vs competitor; ranges over the six shapes):

| op (fp16) | vs eager composite | vs torch.compile'd composite | peak-BW* |
|---|---|---|---|
| `fused_add_rms_norm` | 1.22–1.58× | 0.80–0.99× (0.95–0.99× at M ≥ 2048) | 62–85 % |
| `fused_add_layer_norm` | 1.00–1.49× | 0.48–1.00× (0.88–1.00× at M ≥ 2048) | 27–85 % |
| `rms_norm` vs `F.rms_norm` | 0.99–1.27× | 0.76–1.25× | 33–115 %* |
| `rms_norm_fp8` (dynamic) | 5.6–8.6× | **1.54–1.81× at M ≥ 2048** (0.84× at 512×1024) | 11–41 % |
| `fused_add_rms_norm_fp8` (dyn.) | 5.7–6.9× | **1.44–1.77× at M ≥ 2048** (0.81× at 512×1024) | 23–67 % |
| training step fwd+bwd (LN / RMS) | 0.44–1.04× / 0.55–1.10× | — | — |

fp32 (same file set): `fused_add_rms_norm` 1.24–1.68× vs eager, 0.85–1.04× vs
compiled; `fused_add_layer_norm` 1.27–1.43× vs eager, 0.51–1.03× vs compiled;
the fp8 ops 2.96–5.81× vs eager (fp32's eager chain is relatively cheaper
than fp16's). \*See the L2 note above for >100 % entries.

The backward rows are the deliberately-published weak spot: the backward
kernels are correctness-first (scalar loads, deterministic two-stage parameter
grads) and a full training step measures 0.44–1.10× of PyTorch's autograd —
parity only at the largest shape. Training through these ops is correct,
gradcheck-verified and bitwise reproducible; making it *fast* is future work,
stated here rather than hidden.
