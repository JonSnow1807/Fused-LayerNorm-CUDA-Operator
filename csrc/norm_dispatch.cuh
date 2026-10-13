// Host-side kernel-selection helpers shared by every launcher: the measured
// vec-vs-scalar policy, the block-size heuristic, and the
// FUSED_LAYERNORM_FORCE_KERNEL debug override. Extracted from
// layernorm_cuda_kernel.cu (v0.3.0); the policy constants and their measured
// basis (committed A100 runs under benchmarks/results/) are unchanged.
#pragma once

#include <cstdlib>
#include <cstring>

#include "norm_reduce.cuh"  // kWarpSize, kMaxThreads

namespace fused_norm {

// Kernel choice (measured on an A100-SXM4-40GB, see benchmarks/results/):
//   * Many rows, vectorisable layout: the vectorised kernel with small
//     (<= 256 thread) blocks. With thousands of one-block rows the grid alone
//     fills every SM, and per-row efficiency (16-byte loads) decides
//     throughput.
//   * Few rows: the scalar kernel with up to 1024 threads per row. With only
//     M <= a-few-hundred blocks the GPU is latency-bound and wide blocks put
//     more threads to work; the vectorised kernel measured slower here for
//     every tested shape.
//   * Everything else: the scalar kernel. "Vectorisable" is decided by the
//     caller (all pointers 16-byte aligned and N divisible by the dtype's
//     vector width — see aligned16 in norm_vec.cuh).
constexpr int64_t kVecMinRows = 256;

// Debug/benchmark override: FUSED_LAYERNORM_FORCE_KERNEL=scalar|vec pins the
// kernel choice (used to produce the committed scalar-baseline results).
// Forcing "vec" still requires the layout to be vectorisable at all. Shared
// by choose_vec (block-per-row grids) and launchers with their own
// eligibility rule (the backward partials kernel), so the 3-mode test matrix
// exercises every kernel family. Read once per process.
inline bool apply_force_kernel_env(bool vec, bool vectorisable) {
  static const char* const force_kernel = std::getenv("FUSED_LAYERNORM_FORCE_KERNEL");
  if (force_kernel != nullptr) {
    if (std::strcmp(force_kernel, "scalar") == 0) vec = false;
    if (std::strcmp(force_kernel, "vec") == 0) vec = vectorisable;
  }
  return vec;
}

// Few-row RMSNorm launches on sm_90 (decode-like batches, M < kVecMinRows): measured on an H200,
// the scalar kernel the policy above picks for these rows is latency-bound on one row per SM, and
// a block with up to 512 threads that reads each 16-byte vector once into registers is faster.
// Used by every RMSNorm forward epilogue alike (plain, fused-add, fp8 static/dynamic), so the
// outputs of the fused/quantised ops stay byte-identical to quantising the plain op's output.
// FUSED_LAYERNORM_FORCE_KERNEL=scalar|vec keeps selecting the original kernel families.
// Shape constants (kSmallMMaxThreads, kSmallMVecs) live in norm_reduce.cuh.

inline bool force_kernel_requested() {
  static const bool requested = [] {
    const char* v = std::getenv("FUSED_LAYERNORM_FORCE_KERNEL");
    return v != nullptr && (std::strcmp(v, "scalar") == 0 || std::strcmp(v, "vec") == 0);
  }();
  return requested;
}

// One vector per thread up to kSmallMMaxThreads threads (power of two, at least one warp).
inline int small_m_block_size(int64_t N, int64_t vec_width) {
  const int64_t nvec = N / vec_width;
  int threads = kWarpSize;
  while (threads < nvec && threads < kSmallMMaxThreads) threads *= 2;
  return threads;
}

// The block size is checked against the kernel's register budget explicitly (every thread
// holds at most kSmallMVecs vectors of the row) instead of relying on small_m_block_size's
// formula; the condition depends on the shape only, so plain, fused-add and fp8 launches of a
// shape always take the same kernel.
inline bool choose_small_m_rows(bool vectorisable, int64_t M, int64_t N, int64_t vec_width,
                                int device_major) {
  if (!vectorisable || device_major != 9 || M >= kVecMinRows || N < 128 || force_kernel_requested()) {
    return false;
  }
  const int64_t threads = small_m_block_size(N, vec_width);
  return threads % kWarpSize == 0 && threads <= kSmallMMaxThreads &&
         N / vec_width <= threads * kSmallMVecs;
}

// Fused-add composite contract: a fused-add launch needs input, residual_in and residual_out
// 16-byte aligned for the vectorised kernels, while the composite plain_norm(residual_out) checks
// residual_out alone. When that difference would move the fused launch to another kernel family
// (another summation order) than the composite, the launcher runs the composite instead:
// residual_out = round(input + residual_in), then the plain launch on residual_out. So
// out == plain_norm(new_residual) holds bitwise for every alignment, and launches whose families
// already agree are unchanged.
inline bool choose_vec(bool vectorisable, int64_t M, int64_t N) {
  const bool vec = vectorisable && (N >= 128) && (M >= kVecMinRows);
  return apply_force_kernel_env(vec, vectorisable);
}

// Block size, always a power of two (keeps the reductions' warp arithmetic
// exact). Scalar path: smallest power of two >= N, clamped to [32, 1024] (a
// fixed 1024-thread block would leave most warps idle for short rows, e.g.
// N = 64 would run 2 useful warps and 30 idle ones through every barrier).
// Vectorised path: about two 16-byte vectors per thread, clamped to
// [64, 256] — 128 threads for fp32 N = 1024 matches PyTorch's
// vectorised-kernel block size, and larger rows saturate with 256 threads;
// bigger blocks measured slower on narrow rows, smaller ones on wide rows.
inline int choose_block_size(bool vec, int64_t N, int64_t vec_width) {
  const int64_t items = vec ? (N / vec_width + 1) / 2 : N;
  const int floor_threads = vec ? 2 * kWarpSize : kWarpSize;
  const int cap = vec ? 256 : kMaxThreads;
  int threads = floor_threads;
  while (threads < items && threads < cap) threads *= 2;
  return threads;
}

}  // namespace fused_norm
