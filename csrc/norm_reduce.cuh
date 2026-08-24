// Block/warp reduction machinery shared by every kernel in this library.
//
// Everything here was extracted verbatim from layernorm_cuda_kernel.cu (v0.3.0,
// whose numerics are covered by the committed A100 test/benchmark runs); only
// the namespace is new. blockReduceMax and blockReduceSum2 are additions for
// the fp8-dynamic epilogue and the backward kernels respectively, following the
// same reduction shape and contract.
#pragma once

#include <cuda_runtime.h>

namespace fused_norm {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// Sum `val` across the 32 lanes of a warp. Every lane receives the full sum.
template <typename acc_t>
__device__ __forceinline__ acc_t warpReduceSum(acc_t val) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

// Sum `val` across the whole thread block.
//
// Requirements / contract:
//   * blockDim.x is a power of two in [32, 1024], so blockDim.x / 32 warps exist and each
//     is complete (the launcher guarantees this).
//   * The RETURN VALUE IS ONLY MEANINGFUL ON WARP 0 (callers use it from thread 0);
//     other warps return garbage.
//   * The function contains one __syncthreads(). Two consecutive calls must be separated by
//     a __syncthreads() by the caller, otherwise a fast warp can overwrite `shared` while
//     warp 0 is still reading the previous reduction. In the norm kernels that barrier is
//     the one that publishes s_mean / s_rstd.
//   * `static __shared__` gives every kernel instantiation its own 32-slot scratch array,
//     sized for the maximum of 1024 / 32 = 32 warps.
template <typename acc_t>
__device__ __forceinline__ acc_t blockReduceSum(acc_t val) {
  static __shared__ acc_t shared[kMaxThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;

  val = warpReduceSum<acc_t>(val);          // 1) reduce inside each warp
  if (lane == 0) shared[wid] = val;         // 2) lane 0 of each warp publishes its partial
  __syncthreads();

  // 3) warp 0 gathers the partial sums (one per warp) and reduces them.
  const unsigned int nwarps = blockDim.x / kWarpSize;   // unsigned: matches threadIdx.x
  val = (threadIdx.x < nwarps) ? shared[lane] : static_cast<acc_t>(0);
  if (wid == 0) val = warpReduceSum<acc_t>(val);
  return val;
}

// Two sums reduced together (one shuffle word more per step, one barrier for
// both). Used by the backward dx kernels, which need (sum(g), sum(g*xhat))
// per row. Same contract as blockReduceSum.
template <typename acc_t>
struct Sum2 {
  acc_t a = 0;
  acc_t b = 0;
};

template <typename acc_t>
__device__ __forceinline__ Sum2<acc_t> warpReduceSum2(Sum2<acc_t> v) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    v.a += __shfl_xor_sync(0xffffffffu, v.a, offset);
    v.b += __shfl_xor_sync(0xffffffffu, v.b, offset);
  }
  return v;
}

template <typename acc_t>
__device__ __forceinline__ Sum2<acc_t> blockReduceSum2(Sum2<acc_t> v) {
  static __shared__ Sum2<acc_t> shared[kMaxThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;

  v = warpReduceSum2<acc_t>(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();

  const unsigned int nwarps = blockDim.x / kWarpSize;
  v = (threadIdx.x < nwarps) ? shared[lane] : Sum2<acc_t>{};
  if (wid == 0) v = warpReduceSum2<acc_t>(v);
  return v;
}

// Max over the block (for the fp8-dynamic per-row amax). Same contract as
// blockReduceSum; the identity is 0 because callers reduce |values| >= 0.
template <typename acc_t>
__device__ __forceinline__ acc_t warpReduceMax(acc_t val) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const acc_t other = __shfl_xor_sync(0xffffffffu, val, offset);
    val = other > val ? other : val;
  }
  return val;
}

template <typename acc_t>
__device__ __forceinline__ acc_t blockReduceMax(acc_t val) {
  static __shared__ acc_t shared[kMaxThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;

  val = warpReduceMax<acc_t>(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  const unsigned int nwarps = blockDim.x / kWarpSize;
  val = (threadIdx.x < nwarps) ? shared[lane] : static_cast<acc_t>(0);
  if (wid == 0) val = warpReduceMax<acc_t>(val);
  return val;
}

// Welford partial aggregate: n elements seen, their mean, and the centred sum
// of squares m2 = sum((x - mean)^2). Merging two aggregates (Chan et al.) is
// exact for the same reason the two-pass form is: no E[x^2] - E[x]^2
// cancellation ever occurs.
template <typename acc_t>
struct Welford {
  acc_t mean = 0;
  acc_t m2 = 0;
  acc_t n = 0;  // acc_t (not int): keeps the merge below branch-light. For float
                // acc_t the count stays exact up to 2^24 elements per row; the
                // final variance divides by the exact int64 N either way.
                // (PyTorch's Welford kernel makes the same trade.)
};

template <typename acc_t>
__device__ __forceinline__ Welford<acc_t> welfordMerge(Welford<acc_t> a, Welford<acc_t> b) {
  const acc_t n = a.n + b.n;
  if (n == 0) return a;  // both empty; avoids 0/0 below
  const acc_t delta = b.mean - a.mean;
  Welford<acc_t> out;
  out.n = n;
  out.mean = a.mean + delta * (b.n / n);
  out.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n);
  return out;
}

// Same reduction shape as blockReduceSum above (and the same contract:
// blockDim.x a power of two in [32, 1024], result valid on warp 0, one
// __syncthreads inside, callers separate consecutive calls with a barrier).
template <typename acc_t>
__device__ __forceinline__ Welford<acc_t> warpReduceWelford(Welford<acc_t> w) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    Welford<acc_t> o;
    o.mean = __shfl_xor_sync(0xffffffffu, w.mean, offset);
    o.m2 = __shfl_xor_sync(0xffffffffu, w.m2, offset);
    o.n = __shfl_xor_sync(0xffffffffu, w.n, offset);
    w = welfordMerge(w, o);
  }
  return w;
}

template <typename acc_t>
__device__ __forceinline__ Welford<acc_t> blockReduceWelford(Welford<acc_t> w) {
  static __shared__ Welford<acc_t> shared[kMaxThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;

  w = warpReduceWelford<acc_t>(w);
  if (lane == 0) shared[wid] = w;
  __syncthreads();

  const unsigned int nwarps = blockDim.x / kWarpSize;
  w = (threadIdx.x < nwarps) ? shared[lane] : Welford<acc_t>{};
  if (wid == 0) w = warpReduceWelford<acc_t>(w);
  return w;
}

}  // namespace fused_norm
