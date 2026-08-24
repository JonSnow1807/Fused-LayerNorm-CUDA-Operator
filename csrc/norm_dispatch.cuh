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

inline bool choose_vec(bool vectorisable, int64_t M, int64_t N) {
  bool vec = vectorisable && (N >= 128) && (M >= kVecMinRows);

  // Debug/benchmark override: FUSED_LAYERNORM_FORCE_KERNEL=scalar|vec pins the
  // kernel choice (used to produce the committed scalar-baseline results).
  // Forcing "vec" still requires the layout to be vectorisable at all.
  static const char* const force_kernel = std::getenv("FUSED_LAYERNORM_FORCE_KERNEL");
  if (force_kernel != nullptr) {
    if (std::strcmp(force_kernel, "scalar") == 0) vec = false;
    if (std::strcmp(force_kernel, "vec") == 0) vec = vectorisable;
  }
  return vec;
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
