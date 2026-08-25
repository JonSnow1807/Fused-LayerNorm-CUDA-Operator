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
> commit it was produced from with a clean working tree, and the full test
> suite (currently 260 tests) is run on hardware before any release. v0.3.0
> (2026‑08‑20) added the first kernel-time measurements; v0.4.0 turned the
> single forward-only kernel into the library described below; v0.4.1 and
> v0.4.2 are post-release audit/fuzz patches whose defects — including this
> repository's own — are itemised in the CHANGELOG rather than paraphrased.

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
| `layer_norm` | ✅ | ✅ | – | – |
| `layer_norm_gelu` | ✅ | fallback | – | – |
| `rms_norm` | ✅ | ✅ | – | ✅ static + dynamic |
| `fused_add_layer_norm` | ✅ | ✅ | ✅ (inference) | – |
| `fused_add_rms_norm` | ✅ | ✅ | ✅ (inference) | ✅ static + dynamic |

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
  indistinguishable from the unfused composite.
* fp8 outputs equal quantising the plain norm output with
  `round_e4m3(clamp(y * (1/scale), ±448))` — the reciprocal multiply is part
  of the byte-level contract; `scale` is the dequant scale
  (`y ≈ out.float() * scale`, the vLLM/TensorRT convention). Static scales
  are read on-device (no host sync; CUDA-graph capturable).
* Parameter gradients use a deterministic two-stage reduction (fixed-chunk
  partials + a fixed-shape `sum`), never atomics: backward is bitwise
  run-to-run reproducible.
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

# Drop-in modules (training works: real CUDA backward since v0.4.0).
block = fln.FusedAddRMSNorm(4096, dtype=torch.float16, device="cuda")
normed, stream = block(x, res)

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

Built and tested with PyTorch 2.13.0+cu129 / CUDA 12.9 on an A100‑SXM4‑40GB;
CI compiles for sm80/sm90 and runs the CPU fallback suite on torch 2.4 and
2.13 (CI has no GPU — it cannot verify kernel numerics; the committed
hardware runs do).

## What has been measured

Method for every claim: correctness gates first, then eager per-call latency,
GPU kernel time from `torch.profiler` (the only number quoted as a kernel
speedup), and CUDA-graph replay, produced by the committed scripts from a
clean clone of the release commit, with SM clocks locked at the boost ceiling
(since v0.4.2 — unlocked clocks moved small-shape kernel times ~30 % between
days; [`docs/methodology.md`](docs/methodology.md) §7 has the numbers).
Competitors include the same composite under `torch.compile(fullgraph=True)`
— beating only eager would be a strawman.

### 2026‑08‑25, A100‑SXM4‑40GB: the op family (v0.4.2 data)

[`benchmarks/results/2026-08-25_a100-40gb_v042_ops/`](benchmarks/results/2026-08-25_a100-40gb_v042_ops/)
(fp16 + fp32; produced from a clean clone of `c24b55a`, `git_dirty=false`,
SM clocks recorded in the JSONs as locked at 1410 MHz; supersedes the
v0.4.1 directory after the fp8 NaN-scale
fix — which costs the fp8-dynamic ops ~2–4 % kernel time — and the switch to
locked clocks; details in that directory's README). Kernel-time ratios, ours
vs competitor, over shapes from 512×1024 to 4096×8192:

| op (fp16) | vs eager composite | vs `torch.compile`'d composite |
|---|---|---|
| `fused_add_rms_norm` | **1.23–1.58×** | 0.96–0.99× at M ≥ 2048 (0.80× at 512×1024) |
| `fused_add_layer_norm` | 1.08–1.49× | 0.88–1.00× at M ≥ 2048 |
| `rms_norm` (vs aten's fused `F.rms_norm`) | 0.98–1.29× | 0.77–1.28× |
| `rms_norm_fp8` dynamic | **4.8–7.0×** | **1.03–1.73× — ≥ 1 at every shape** |
| `fused_add_rms_norm_fp8` dynamic | **4.9–5.9×** | **1.01–1.74× — ≥ 1 at every shape** |
| training step fwd+bwd (LN / RMS) | 0.43–1.04× / 0.55–1.10× | — |

How to read it honestly:

* **The fused-add ops deliver what they promise in eager mode**: the RMS op
  at 1.23–1.58× kernel time over the eager composite (63–85 % of datasheet
  bandwidth), the LayerNorm op at 1.08–1.49× (35–84 %), both at kernel parity
  with Inductor's fused codegen at production shapes. On **wall clock** they
  beat the eager composite everywhere (1.12–1.48×, both dtypes) and the
  compiled composite at most shapes — it pays ~90 µs of guard/dispatch per
  eager call (2048×4096 fp16: ours 55 µs, compiled 1.64× slower) — but ties
  it at fp16's two largest shapes (0.99–1.01×) and **loses to it by ~15 % at
  fp32's largest shape** (0.85–0.86× at 4096×8192): once the guard overhead
  is amortised over a big enough call, Inductor's kernels win that one.
* **The norm→fp8 fusion is the headline**: 4.8–7.0× over the eager
  norm→amax→cast chain, and the one place this library beats
  `torch.compile`'s fused kernel outright — **≥ 1× at every measured shape**
  (up to 1.74×): Inductor fuses the chain but still materialises
  intermediates the kernel keeps in registers/L1. Full disclosure: at
  512×1024 the fp16 lead is now 1.01–1.03× (fp32: 1.19–1.21×) — a few points
  of it went to the v0.4.2 NaN-correctness fix, a trade this repository
  makes without hesitation.
* Plain `rms_norm` competes with aten's own fused kernel and still comes out
  ahead at most shapes (up to 1.29× fp16, 1.32× fp32; ~93 % of peak at
  4096×4096) — but near-parity, not headlines, is the honest framing there.
* **The published weak spots**: at 512×1024 Inductor's small-shape kernels
  beat ours on pure kernel time for the non-fp8 ops (0.52–0.95× across both
  dtypes — while every compiled candidate pays 4.1–10.4× more wall clock per
  call at that shape); fp32's largest shape loses to the compiled composite
  on wall clock (above); and a full training step measures **0.43–1.10× of
  PyTorch's autograd in fp16** (fp32: 0.53–1.28×,
  beating autograd at the three largest shapes) — the backward kernels are
  correctness-first (scalar loads, deterministic no-atomics parameter grads,
  gradcheck-verified). Training through these ops is correct and bitwise
  reproducible; making it fast is future work.
* A few JSON rows show >100 % of datasheet bandwidth. The bytes models count
  each tensor exactly once, so the real mechanism is inter-call caching: at
  those shapes the working set (8–34 MB) fits the A100's 40 MB L2 and stays
  resident across the 200 timed calls, so much of the "DRAM" traffic never
  leaves L2. Compiled candidates exceed 100 % at the same shapes, and
  candidate *ratios* are unaffected (same model both sides).

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

* One GPU (A100). The kernel-selection heuristics are tuned on A100
  measurements; other architectures compile but are unmeasured.
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
* `layer_norm_gelu` remains forward-only (falls back under grad).
* fp8 ops are inference-only by design; sm_80 emulates the fp8 convert in
  software (no hardware convert before sm_89) — measured, not assumed.
* Under CUDA autocast every op defers to PyTorch (keeps autocast's fp32
  output semantics exactly).

## Project layout

```
csrc/                       kernels: layernorm_cuda_kernel.cu (v0.3.0 LN kernels, unchanged),
                            norm_fwd_kernels.cuh (generic {LN,RMS} x {plain,fused-add} x epilogue),
                            norm_fwd_{ln,rms}.cu, norm_bwd.cu, norm_{reduce,vec,epilogue,dispatch}.cuh,
                            bindings.cpp, layernorm.h
fused_layernorm/            package: layernorm.py, rms_norm.py, fused_add.py, quant.py,
                            _ops.py (torch.library custom ops + autograd), _common.py
tests/                      260 tests: per-op suites, gradcheck (fp64), bitwise contracts,
                            opcheck, torch.compile fullgraph, determinism, NaN semantics;
                            fuzz_contracts.py (randomized contract fuzz, run explicitly)
benchmarks/                 bench_layernorm.py (LayerNorm), bench_norms.py (v0.4.0 op family),
                            results/ (committed data with git provenance), legacy/
docs/methodology.md         how to time a small CUDA op without fooling yourself
.github/workflows/ci.yml    CPU test matrix + CUDA compile-only job
CHANGELOG.md                every release, including what was once claimed falsely and why
```

## Contributing

Most useful: run the benchmarks and test suite on a non-A100 GPU and open a
PR with the JSON. Also welcome: GELU backward, Hopper tuning, int8 epilogue.

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
