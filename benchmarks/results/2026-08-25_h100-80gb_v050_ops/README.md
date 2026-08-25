# 2026-08-25 — H100 80GB HBM3 — v0.5.0 measurements (second-GPU validation)

The first non-A100 data in the repository. Same code as
`../2026-08-25_a100-40gb_v050_ops/` (clean clone of `8af8dbb`, the v0.5.0
release), same script, same method; extension built with
`TORCH_CUDA_ARCH_LIST=9.0`. JSON metadata: `git_dirty = false`,
`extension_version = 0.5.0`, driver 570.148.08, 3 350 GB/s datasheet peak,
`gpu_clock_state` showing SM clocks locked at **1830 MHz** with persistence
on — 1830, not the 1980 boost ceiling, because 1830 is what this GPU
sustains under memory-bound norm load at ~590 W (a dense-matmul load dips
to ~1600); pinning at the sustainable clock keeps every shape at one
constant frequency (methodology §7). The 289-test suite passed in the same
clone in all three kernel modes, plus the randomized contract fuzz — the
kernels are **correct on sm90**, including the fp8 converts, which are
native hardware instructions here (the A100 emulates them).

## Read this first: what transfers from the A100 data and what does not

**Transfers — the eager-mode wins, which are what the library is for:**

| vs the EAGER composite (kernel time) | fp16 | fp32 |
|---|---|---|
| `fused_add_rms_norm` | 1.19–1.57× | 1.24–1.70× |
| `fused_add_layer_norm` | 1.02–1.44× | 1.25–1.42× |
| `rms_norm` vs `F.rms_norm` | 1.02–1.28× | 1.04–1.15× |
| `rms_norm_fp8` (dynamic) | **7.7–11.4×** | 5.4–7.1× |
| `fused_add_rms_norm_fp8` (dyn.) | **6.3–7.7×** | 3.5–6.7× |
| `layer_norm_fp8` (dynamic) | 3.8–7.7× | 3.6–5.4× |
| `fused_add_layer_norm_fp8` (dyn.) | 3.8–6.6× | 3.4–4.6× |
| training step LN / RMS / LN+GELU | 0.69–1.42× / 0.84–1.79× / 0.79–1.13× | 0.77–1.28× / 0.86–1.28× / 0.94–1.56× |

The fp8-vs-eager margins are LARGER than on the A100 (native converts, and
the eager chain wastes more of a faster GPU); the fused-add and backward
stories hold (training ≥ 1× autograd at most M ≥ 2048 shapes, RMS up to
1.79×); bandwidth utilisation reaches 87 % of the H100 datasheet peak on
the fused-add forwards.

**Does NOT transfer — the kernel-time edge over `torch.compile`:**

On the A100, the RMS fp8 ops beat the compiled chain at every shape and the
fused-add ops held 0.96–0.99× parity. On the H100 that is not true:
Inductor's Hopper codegen is markedly stronger while these kernels' launch
heuristics (vector widths, block sizes, `choose_vec` thresholds) are tuned
from A100 measurements only. Measured vs the compiled composite, kernel
time: `rms_norm_fp8` 0.70–1.20× fp16 (0.70–1.05× fp32),
`fused_add_rms_norm_fp8` 0.69–1.06× (0.76–1.06×), the LN fp8 ops
0.35–1.27×, `fused_add_rms_norm` 0.70–0.99×, `fused_add_layer_norm`
0.42–1.04×, `rms_norm` 0.67–1.07×. **No "≥ 1× vs compiled" claim is made
for the H100.** On wall clock the picture is friendlier (compiled pays its
per-call dispatch everywhere): the RMS fp8 ops still beat the compiled
chain at every measured fp16 shape (1.00–3.40×), the other ops at most
shapes, with sub-1× cells at the largest/tallest shapes (worst 0.74×).
Hopper-specific tuning is the top item on the contributing list; until it
exists, the A100 claims are A100 claims.

A few rows exceed 100 % of the datasheet peak (fp16 `rms_norm` at
2048×4096 and 8192×1024): same inter-call L2-residency mechanism as on the
A100 — the H100's 50 MB L2 holds the 8–34 MB working sets across the 200
timed calls; compiled candidates show the same effect and ratios are
unaffected.
