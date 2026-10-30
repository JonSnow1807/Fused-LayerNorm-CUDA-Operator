# fused_layernorm — fused normalisation kernels for PyTorch

[![CI](https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator/actions/workflows/ci.yml/badge.svg)](https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator/actions/workflows/ci.yml)

LayerNorm and RMSNorm CUDA kernels with the fusions eager PyTorch does not
have: **fused residual-add + norm** (the pattern that runs twice per pre-norm
transformer layer), **fp8-E4M3 quantised outputs** (static and dynamic
per-token scales), a **real CUDA backward** for training, and
**`torch.compile` integration without graph breaks** — behind drop-in
`F.layer_norm` / `F.rms_norm`-shaped functions and `nn.LayerNorm` /
`nn.RMSNorm` subclasses, with an honest, reproducible account of every
measured claim.

> **Provenance ledger.** This repository was rewritten for accuracy on
> 2026‑08‑18 after an audit found its earlier claims matched no committed data
> ([`CHANGELOG.md`](CHANGELOG.md) lists everything removed). Since then every
> number quoted here points at a committed file under
> [`benchmarks/results/`](benchmarks/results/) whose JSON records the git
> commit it was produced from with a clean working tree (one exception, the
> 2026‑10‑29 H100 NVL A/B run, records `git_dirty = true` because it wrote
> its output inside the checkout; its README says so), and the full test
> suite (currently 335 tests) is run on hardware before any release. v0.3.0
> (2026‑08‑20) added the first kernel-time measurements; v0.4.0 turned the
> single forward-only kernel into the library described below; v0.4.1 and
> v0.4.2 are post-release patches for defects found by re-auditing and
> fuzzing our own release — each itemised in the CHANGELOG rather than
> paraphrased; v0.6.0 (2026‑10‑29) adds two Hopper RMSNorm forward paths
> and fixes the fused-add alignment contract.

## Why this exists

Eager PyTorch has no fused `residual = x + residual; y = norm(residual)` op —
`aten` simply doesn't contain one — so eager transformer code pays a full
extra HBM round-trip of the hidden state at every norm site.
`torch.compile` closes most of that gap when you can use it; these kernels
close it in eager mode too, keep working under `torch.compile` (they trace as
single custom-op nodes), and add the norm→fp8 fusion that serving stacks want
before fp8 GEMMs. Where PyTorch is already optimal, this README says so
instead of benchmarking around it.

## The ops

| op | fwd | bwd (training) | inplace | fp8 out |
|---|---|---|---|---|
| `layer_norm` | ✅ | ✅ | – | ✅ static + dynamic |
| `layer_norm_gelu` | ✅ | ✅ (erf + tanh) | – | – |
| `rms_norm` | ✅ | ✅ | – | ✅ static + dynamic |
| `fused_add_layer_norm` | ✅ | ✅ | ✅ (inference) | ✅ static + dynamic |
| `fused_add_rms_norm` | ✅ | ✅ | ✅ (inference) | ✅ static + dynamic |

Since v0.5.0 the table has no asterisks left: every op has a real CUDA
backward (vectorised, deterministic — see below), and fp8 outputs cover
both norm families.

Modules: `LayerNorm(nn.LayerNorm)`, `RMSNorm(nn.RMSNorm)` (exact drop-ins,
including `F.rms_norm`'s subtle `eps=None` = machine-epsilon semantics),
`FusedAddLayerNorm` / `FusedAddRMSNorm` (pre-norm blocks returning
`(normed, new_residual)`), and per-model `replace_layernorm` /
`replace_rmsnorm` helpers (exact-type match, shared parameters, nothing
monkeypatched). Dtypes: fp32/fp16/bf16 everywhere, fp64 on the non-quantised
paths (it exists so `torch.autograd.gradcheck` can run). torch ≥ 2.4.

**Contracts worth knowing** (all tested bit-for-bit):

* Fused-add computes `new_residual = round(x + residual)` — rounded to the
  input dtype exactly once — and normalises the *rounded* sum, so
  `out == plain_norm(new_residual)` holds **bitwise**: the fused op is
  indistinguishable from the unfused composite. This holds for any input
  alignment: if a contiguous but misaligned `x` or `residual` (a `base[1:]`
  view) would put the fused call on a different kernel than the composite,
  the call runs the composite itself (the add, then the plain norm).
* fp8 outputs equal quantising the plain norm output with
  `round_e4m3(clamp(y * (1/scale), ±448))` — the reciprocal multiply is part
  of the byte-level contract; `scale` is the dequant scale
  (`y ≈ out.float() * scale`, the vLLM/TensorRT convention). Static scales
  are read on-device (no host sync; CUDA-graph capturable).
* Parameter gradients use a deterministic two-stage reduction (fixed-chunk
  partials + a fixed-order finalize kernel), never atomics: backward is
  bitwise run-to-run reproducible.
* `inplace=True` mutates only the residual, requires it contiguous (no
  silent copy), and is inference-only (raises under grad).
* NaN never disappears: a NaN input yields NaN fp8 bytes **and** a NaN
  dynamic scale (`torch.amax` semantics — the amax reduction propagates NaN
  instead of dropping it as `fmaxf` would), so a poisoned row can't
  masquerade as tiny finite values after dequantisation.

## Quickstart

```python
import torch, fused_layernorm as fln

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
res = torch.randn_like(x)
w = torch.rand(4096, device="cuda", dtype=torch.float16) + 0.5

# The headline op: one kernel instead of add + norm.
out, new_res = fln.fused_add_rms_norm(x, res, (4096,), w)

# Drop-in modules. Training works: vectorised, deterministic CUDA backward
# (>= 1x autograd kernel time at M >= 2048 shapes since v0.5.0; A100).
block = fln.FusedAddRMSNorm(4096, dtype=torch.float16, device="cuda")
normed, new_res2 = block(x, res)

# norm -> fp8 in one kernel (inference; dynamic per-token scales).
q, scale = fln.rms_norm_fp8(x, (4096,), w)     # q: float8_e4m3fn
y = q.float() * scale                           # dequantise

# torch.compile: traces as single custom-op nodes, zero graph breaks.
compiled = torch.compile(lambda x, r: fln.fused_add_rms_norm(x, r, (4096,), w),
                         fullgraph=True)
```

Every function falls back to the equivalent PyTorch composite when the fused
kernel does not apply (CPU tensors, autocast regions, mismatched dtypes,
multi-dim `normalized_shape`, missing extension), so behaviour — output
dtypes and errors included — matches PyTorch everywhere.

## Install

Requirements: a CUDA GPU, a CUDA-enabled PyTorch ≥ 2.4, and the matching CUDA
toolkit with `nvcc` on `PATH`.

```bash
git clone https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator.git
cd Fused-LayerNorm-CUDA-Operator
pip install --no-build-isolation -e .   # builds against the torch you have installed
python -m pytest tests -q               # GPU tests skip without a GPU/extension
```

Built and tested with PyTorch 2.13.0+cu129 / CUDA 12.9 on an A100‑SXM4‑40GB
and with 2.13.0+cu130 / nvcc 13.4 on an H100 NVL (v0.6.0);
CI compiles for sm80/sm90 and runs the CPU fallback suite on torch 2.4 and
2.13 (CI has no GPU — it cannot verify kernel numerics; the committed
hardware runs do).

## What has been measured

Method for every claim: correctness gates first, then eager per-call latency,
GPU kernel time from `torch.profiler` (the only number quoted as a kernel
speedup), and CUDA-graph replay, produced by the committed scripts from a
clean clone of the release commit, with SM clocks locked (since v0.4.2 —
at the boost ceiling on the A100, at the sustained clock on the H100;
unlocked clocks moved small-shape kernel times ~30 % between days; [`docs/methodology.md`](docs/methodology.md) §7 has the numbers.
The one run that could not lock clocks, the 2026‑10‑29 H100 NVL A/B, says
so and quotes only same-session ratios).
Competitors include the same composite under `torch.compile(fullgraph=True)`
— beating only eager would be a strawman.

### 2026‑10‑29, H100 NVL: the v0.6.0 Hopper RMSNorm fast paths vs v0.5.0

[`benchmarks/results/2026-10-29_h100-nvl_v060_fastpaths/`](benchmarks/results/2026-10-29_h100-nvl_v060_fastpaths/)
(fp16 + bf16 + fp32; the PR #2 branch at `5268275` against `origin/main`
`93d5882`, whose forward kernels are the v0.5.0 tag's; one container, the
same `bench_norms.py` invocation for both builds, 4 RMSNorm ops × 18
shapes; torch 2.13.0+cu130, nvcc 13.4; SM clock **not lockable** there —
1785 MHz recorded at the start of every run; `git_dirty = true` for the
reason the directory README gives). v0.6.0 adds two sm_90 forward paths: a
register-resident **few-row kernel** for RMSNorm calls with fewer than 256
rows, and a **warp-per-row dynamic-FP8 kernel** for hidden sizes ≤ 1024
(fp16/bf16; ≤ 512 fp32) at 256+ rows. Kernel time of the branch as a
fraction of v0.5.0's (below 1 = faster), min / median / max over every row
the path dispatches on:

| path | fp16 | bf16 | fp32 |
|---|---|---|---|
| few-row kernel (48 / 48 / 44 rows) | 0.40 / 0.65 / 1.00 | 0.42 / 0.67 / 1.02 | 0.54 / 0.74 / 0.96 |
| warp-per-row dynamic FP8 (3 / 3 / 0 rows; hidden size 1024 only) | 0.70 / 0.89 / 0.95 | 0.67 / 0.90 / 0.91 | not measured |
| control: same kernel in both builds (70 rows, all dtypes) | 0.96 / 1.00 / 1.05 | | |

How to read it honestly:

* **The few-row kernel is faster at every measured shape of width ≥ 4096,
  in all three dtypes** (116 rows, worst 0.92×): per-op medians 0.62–0.66×
  fp16, 0.64–0.70× bf16, 0.65–0.85× fp32 — e.g. `rms_norm` bf16 64×4096
  3.41 → 2.37 µs, fp32 64×4096 3.29 → 2.35 µs, bf16 32×16384
  8.59 → 3.62 µs. At 32×768 and 32×1024 it is a wash (0.83–1.02×; the one
  row above 1, `rms_norm` bf16 32×768 at 1.02×, is inside the ±4 %
  control-row noise). Graph-replay medians 0.69 / 0.71 / 0.79.
* **The warp-per-row FP8 kernel has H100 data at hidden size 1024 only**:
  `rms_norm_fp8_dynamic` 8192×1024 bf16 16.81 → 11.30 µs (0.67×),
  512×1024 0.89–0.90×, the fused-add variant 0.91–0.95× at 8192×1024. Its
  N ≤ 512 and fp32 ranges have no committed measurement; the PR's H200
  numbers (−8 % to −51 % kernel time over the whole range) are the only
  data there and are quoted in the CHANGELOG as such.
* **Nothing else moved**: the 70 rows that run the same kernel in both
  builds sit at a 1.00 median, 0.96–1.05, which is also the noise floor for
  every number above. One run per cell; branch and baseline not
  interleaved (22:18–22:24 Z vs 22:32–22:36 Z); absolute microseconds are
  not comparable to the H200 or to the 2026‑08‑25 H100 80GB HBM3 data.

### 2026‑08‑25, A100‑SXM4‑40GB: the op family (v0.5.0 data)

[`benchmarks/results/2026-08-25_a100-40gb_v050_ops/`](benchmarks/results/2026-08-25_a100-40gb_v050_ops/)
(fp16 + fp32; produced from a clean clone of `b38a935`, `git_dirty=false`,
SM clocks recorded in the JSONs as locked at 1410 MHz; supersedes the
v0.4.2 directory after the backward kernels were vectorised and the GELU
backward and LN-family fp8 ops landed; details and full tables in that
directory's README). Kernel-time ratios, ours vs competitor, over shapes
from 512×1024 to 4096×8192:

| op (fp16) | vs eager composite | vs `torch.compile`'d composite |
|---|---|---|
| `fused_add_rms_norm` | **1.23–1.59×** | 0.96–0.99× at M ≥ 2048 (0.81× at 512×1024) |
| `fused_add_layer_norm` | 1.08–1.49× | 0.88–1.00× at M ≥ 2048 |
| `rms_norm` (competitor: aten's fused `F.rms_norm`) | 0.99–1.29× | 0.77–1.26× |
| `rms_norm_fp8` dynamic | **4.9–7.2×** | **1.04–1.76× — ≥ 1 at every shape** |
| `fused_add_rms_norm_fp8` dynamic | **5.0–6.0×** | **1.01–1.76× — ≥ 1 at every shape** |
| `layer_norm_fp8` dynamic | 3.5–6.0× | 1.02–1.49× at M ≥ 2048 (0.75× at 512×1024) |
| `fused_add_layer_norm_fp8` dynamic | 3.6–5.7× | 1.25–1.68× at M ≥ 2048 (0.74× at 512×1024) |
| training step fwd+bwd vs autograd (LN / RMS / LN+GELU) | 0.75–1.42× / 0.82–1.44× / 0.80–1.44× | — |

How to read it honestly:

* **Training through these ops is now fast, not just correct** (the v0.5.0
  change): the backward kernels use the same 16-byte vectorised loads as
  the forwards, a GPU-filling chunk policy and a single fused stage-2
  reduction — still deterministic, no atomics, bitwise run-to-run
  reproducible. A full fwd+bwd LN/RMS step measures **1.17–1.44× of
  PyTorch's autograd at every M ≥ 2048 shape in fp16** (GELU step:
  1.16–1.44×; fp32: 1.01–1.16×, and the fused GELU step 1.51–1.88× —
  autograd pays dearly for the erf chain). The
  remaining sub-1× cell is 512×1024 kernel time (0.75–0.86×) plus
  dispatch-bound small-M wall clock (~0.5×) — both engine/launch-bound,
  published, and amortised away by `torch.compile` or larger batches.
* **The fused-add ops deliver what they promise in eager mode**: the RMS op
  at 1.23–1.59× kernel time over the eager composite (64–86 % of datasheet
  bandwidth), the LayerNorm op at 1.08–1.49× (35–84 %), both within 0.88-1.00x of
  Inductor's fused codegen at production shapes (M ≥ 2048). On **wall clock** they
  beat the eager composite everywhere (1.13–1.48×, both dtypes) and the
  compiled composite at most shapes — it pays ~90 µs of guard/dispatch per
  eager call — but ties it at fp16's largest shapes (0.99–1.01×) and **loses
  to it by ~15 % at fp32's largest shape** (0.85–0.86× at 4096×8192): once
  the guard overhead is amortised over a big enough call, Inductor's kernels
  win that one.
* **The norm→fp8 fusion is the headline**: 3.5–7.2× over the eager
  norm→amax→cast chain across both families, and the RMS fp8 ops beat
  `torch.compile`'s fused kernel outright — **≥ 1× at every measured shape
  and dtype** (up to 1.76×). That claim is deliberately RMS-only: the new
  LayerNorm fp8 ops win at every M ≥ 2048 shape (1.02–1.68×) but **lose the
  smallest shape to Inductor** (0.74–0.88× at 512×1024). On wall clock every
  fp8 op beats the compiled chain at every measured shape (1.08–10.8×).
* Plain `rms_norm` competes with aten's own fused kernel and still comes out
  ahead at most shapes (up to 1.29× fp16, 1.31× fp32; ~96 % of peak at
  4096×4096) — but near-parity, not headlines, is the honest framing there.
* **The published weak spots**: 512×1024 — Inductor's small-shape kernels
  beat ours on pure kernel time for the non-fp8 forwards (0.52–0.95× across
  both dtypes, while every compiled candidate pays 4.2–10.8× more wall clock
  per call at that shape) and the training step is 0.75–0.95× there; and
  fp32's largest shape loses to the compiled composite on wall clock
  (above).
* A few JSON rows show >100 % of datasheet bandwidth. The bytes models count
  each tensor exactly once, so the real mechanism is inter-call caching: at
  those shapes the working set (8–34 MB) fits the A100's 40 MB L2 and stays
  resident across the 200 timed calls, so much of the "DRAM" traffic never
  leaves L2. Compiled candidates exceed 100 % at the same shapes, and
  candidate *ratios* are unaffected (same model both sides).

### 2026‑08‑25, H100 80GB HBM3: second-GPU validation (v0.5.0 data)

[`benchmarks/results/2026-08-25_h100-80gb_v050_ops/`](benchmarks/results/2026-08-25_h100-80gb_v050_ops/)
(same v0.5.0 code, cloned clean at `8af8dbb`, built for sm90, clocks locked at the sustained
1830 MHz, full suite + fuzz green — the kernels are **correct on Hopper**,
with native fp8 converts). Two findings, both published as measured:

* **The eager-mode wins transfer and grow**: fused-add ops 1.0–1.7× over
  the eager composite, `rms_norm_fp8` **7.7–11.4×** over the eager chain in
  fp16 (native converts; A100: 4.9–7.2×), training steps up to 1.79× of
  autograd, up to 87 % of the 3 350 GB/s datasheet peak.
* **The kernel-time edge over `torch.compile` does NOT transfer**: Inductor
  generates markedly better Hopper kernels while these launch heuristics
  are tuned from A100 measurements only — the RMS fp8 ops measure
  0.69–1.20× vs the compiled chain there (A100: ≥ 1× everywhere), so **the
  "≥ 1× vs compiled" claims in the table above are A100 claims**. On wall
  clock the RMS fp8 ops still beat the compiled chain at every fp16 shape
  (1.00–3.40×). Hopper-specific tuning is the top item on the contributing
  list; the H100
  directory's README has the full honest breakdown.

### 2026‑08‑20, A100‑SXM4‑40GB: LayerNorm kernel time (v0.3.0)

[`benchmarks/results/2026-08-20_a100-40gb_kernel_time/`](benchmarks/results/2026-08-20_a100-40gb_kernel_time/):
the plain-LayerNorm kernels (unchanged in v0.4.0) measured against
`F.layer_norm`. fp32 kernel time: faster on 9 of 11 shapes (1.01–1.97×),
parity at 8192×1024, 0.93× at 512×1024; fp16: faster everywhere (1.02–1.98×)
except 512×1024 (0.74×); scalar-baseline and full tables in that directory
and in [`CHANGELOG.md`](CHANGELOG.md). The August‑2025 eager-latency data and
its history live in
[`benchmarks/results/2025-08-17_a100_eager_latency/`](benchmarks/results/2025-08-17_a100_eager_latency/).

## Not verified / limitations

* Three GPUs measured (A100-SXM4-40GB, H100 80GB HBM3, H100 NVL); other
  architectures compile but are unmeasured. The v0.5.0 kernel-selection
  heuristics are tuned on A100 measurements — on the H100 the kernels are
  correct and keep their eager-mode wins, but the kernel-time edge over
  `torch.compile` is A100-specific (measured and published in the H100
  results directory). v0.6.0's two sm_90 RMSNorm paths were measured
  against v0.5.0 on an H100 NVL (committed) and on an H200 (PR #2, not
  committed); the H100 NVL files contain `torch.compile` candidates but no
  claim against `torch.compile` is made from them.
* The warp-per-row dynamic-FP8 path has committed data at hidden size 1024
  (fp16/bf16) only; its N ≤ 512 and fp32 ranges rest on the PR's H200
  numbers.
* The few-row kernel changes the fp32 summation order below 256 rows. Over
  16,800 in-target fp32 cases on an H100 its per-case max error stayed
  within the v0.5.0 kernel's plus 2^-21 (one fp32 ulp at |y| in [4, 8); the
  largest excess was exactly that), and it was worse in 2435 cases and
  better in 2489 — not systematically worse. The exact rule and what it
  does not say are in the 0.6.0 CHANGELOG entry.
* Small-shape (~10 µs) kernel times are properties of the GPU's clock state
  as much as of the kernel; committed runs lock SM clocks (methodology §7),
  and claims that live on a few percent at those shapes should be read with
  that in mind.
* Plain `rms_norm` forward competes with aten's own fused kernel — parity or
  modest wins, not headlines; the fused-add and fp8 ops are where the eager
  gap is.
* At small shapes Inductor's generated fused kernel can beat ours on pure
  kernel time (while paying far higher wall-clock latency); the committed
  tables show both numbers.
* Training steps at M ≲ 1024 remain below autograd on kernel time
  (0.75–0.95× at 512×1024) and dispatch-bound on wall clock — both sides
  are launch/engine-bound there and ours pays more launches.
* fp8 ops are inference-only by design; sm_80 emulates the fp8 convert in
  software (no hardware convert before sm_89) — measured, not assumed. The
  LayerNorm-family fp8 ops lose the smallest measured shape to Inductor on
  kernel time (the RMS family does not).
* Under CUDA autocast every op defers to PyTorch (keeps autocast's fp32
  output semantics exactly).

## Project layout

```
csrc/                       kernels: layernorm_cuda_kernel.cu (v0.3.0 LN kernels, unchanged),
                            norm_fwd_kernels.cuh (generic {LN,RMS} x {plain,fused-add} x epilogue,
                            plus the v0.6.0 sm_90 RMSNorm few-row and warp-per-row fp8 kernels),
                            norm_fwd_{ln,rms}.cu, norm_bwd{,_gelu}.cu, norm_bwd_kernels.cuh,
                            norm_{reduce,vec,epilogue,dispatch}.cuh, bindings.cpp, layernorm.h
fused_layernorm/            package: layernorm.py, rms_norm.py, fused_add.py, quant.py,
                            _ops.py (torch.library custom ops + autograd), _common.py
tests/                      335 tests: per-op suites, gradcheck (fp64), bitwise contracts,
                            opcheck, torch.compile fullgraph, determinism, NaN semantics;
                            fuzz_contracts.py (randomized contract fuzz, run explicitly)
benchmarks/                 bench_layernorm.py (LayerNorm), bench_norms.py (the op family),
                            results/ (committed data with git provenance), legacy/
docs/methodology.md         how to time a small CUDA op without fooling yourself
.github/workflows/ci.yml    CPU test matrix + CUDA compile-only job
CHANGELOG.md                every release, including what was once claimed falsely and why
```

## Contributing

Most useful: run the benchmarks and test suite on a non-A100 GPU and open a
PR with the JSON. Also welcome: Hopper tuning, int8 epilogue, small-batch
training-step latency work.

## Citation

```bibtex
@software{shrivastava2026fused_layernorm,
  title  = {fused_layernorm: fused normalisation CUDA kernels for PyTorch},
  author = {Chinmay Shrivastava},
  year   = {2026},
  url    = {https://github.com/JonSnow1807/Fused-LayerNorm-CUDA-Operator}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
