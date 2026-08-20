// Fused LayerNorm (+ optional GELU) forward kernel for PyTorch.
//
// Design (deliberately simple; readability is the point of this repository):
//   * One thread block per row. The row of length N is normalised over its last dimension.
//   * Two-pass statistics: first the mean, then the centred sum of squares. Two passes over
//     the row cost one extra read of data that is already in L1/L2, and are numerically
//     sound (no catastrophic cancellation of E[x^2] - E[x]^2).
//   * All arithmetic is done in acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>
//     (float for float/half/bfloat16, double for double); only the final store casts back.
//   * Forward only. There is no backward kernel and no autograd support.
//
// The launcher at the bottom is the only host-side entry point (declared in layernorm.h).
// Argument validation and reshaping to 2-D happen in bindings.cpp, not here.

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>       // at::acc_type
#include <ATen/cuda/CUDAContext.h>     // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>    // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <c10/cuda/CUDAGuard.h>        // c10::cuda::CUDAGuard

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "layernorm.h"

namespace {

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
//   * The RETURN VALUE IS ONLY MEANINGFUL ON WARP 0 (and is used from thread 0 below);
//     other warps return garbage.
//   * The function contains one __syncthreads(). Two consecutive calls must be separated by
//     a __syncthreads() by the caller, otherwise a fast warp can overwrite `shared` while
//     warp 0 is still reading the previous reduction. In this kernel that barrier is the one
//     that publishes s_mean / s_rstd.
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

// GELU, exact form:  0.5 * x * (1 + erf(x / sqrt(2))).
// The literal is 1/sqrt(2). Called unqualified so the float / double overloads of erf resolve
// for acc_t = float / double respectively.
template <typename acc_t>
__device__ __forceinline__ acc_t gelu_erf(acc_t x) {
  const acc_t kInvSqrt2 = static_cast<acc_t>(0.70710678118654752440);
  return static_cast<acc_t>(0.5) * x * (static_cast<acc_t>(1) + erf(x * kInvSqrt2));
}

// GELU, tanh approximation:  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// The literal 0.7978845608... is sqrt(2/pi). Same constants as PyTorch's approximate="tanh".
template <typename acc_t>
__device__ __forceinline__ acc_t gelu_tanh(acc_t x) {
  const acc_t kBeta = static_cast<acc_t>(0.79788456080286535588);
  const acc_t kKappa = static_cast<acc_t>(0.044715);
  const acc_t inner = kBeta * (x + kKappa * x * x * x);
  return static_cast<acc_t>(0.5) * x * (static_cast<acc_t>(1) + tanh(inner));
}

// The kernel. One block handles one row of N elements.
//   input / output : contiguous (M, N)
//   gamma / beta   : length-N or nullptr; each is applied independently when present
//   N              : row length (int64_t so that row * N cannot overflow 32 bits)
//   eps            : already converted to acc_t by the launcher
//   use_tanh       : selects the GELU variant; ignored when kGelu is false
template <typename scalar_t, typename acc_t, bool kGelu>
__global__ void layernorm_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const scalar_t* __restrict__ gamma,
                                 const scalar_t* __restrict__ beta,
                                 int64_t N,
                                 acc_t eps,
                                 bool use_tanh) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  const scalar_t* X = input + row * N;   // 64-bit offset
  scalar_t* Y = output + row * N;

  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;

  // Pass 1: mean. Each thread sums a strided slice of the row, then the block reduces.
  acc_t sum = 0;
  for (int64_t i = tid; i < N; i += stride) {
    sum += static_cast<acc_t>(X[i]);
  }
  sum = blockReduceSum<acc_t>(sum);
  if (tid == 0) s_mean = sum / static_cast<acc_t>(N);
  __syncthreads();  // publishes s_mean AND separates the two blockReduceSum calls
  const acc_t mean = s_mean;

  // Pass 2: biased variance around the mean, then rstd = 1 / sqrt(var + eps).
  // rsqrt is called unqualified so the float / double device overloads resolve for acc_t.
  acc_t sq = 0;
  for (int64_t i = tid; i < N; i += stride) {
    const acc_t d = static_cast<acc_t>(X[i]) - mean;
    sq += d * d;
  }
  sq = blockReduceSum<acc_t>(sq);
  if (tid == 0) s_rstd = rsqrt(sq / static_cast<acc_t>(N) + eps);
  __syncthreads();
  const acc_t rstd = s_rstd;

  // Pass 3: normalise, optional affine, optional GELU, store.
  for (int64_t i = tid; i < N; i += stride) {
    acc_t v = (static_cast<acc_t>(X[i]) - mean) * rstd;
    if (gamma) v *= static_cast<acc_t>(gamma[i]);
    if (beta) v += static_cast<acc_t>(beta[i]);
    if (kGelu) v = use_tanh ? gelu_tanh<acc_t>(v) : gelu_erf<acc_t>(v);
    Y[i] = static_cast<scalar_t>(v);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Vectorised single-pass variant (used when N is a multiple of the 16-byte
// vector width, N >= 128 and there are >= 256 rows; see the launcher).
//
// Two changes relative to layernorm_kernel above, both copied from what
// PyTorch's own vectorized_layer_norm_kernel does:
//   * 16-byte aligned vector loads/stores (kVecWidth elements: 4 x float,
//     8 x half/bfloat16, 2 x double). The launcher takes this path only when
//     N % kVecWidth == 0 AND every data pointer is 16-byte aligned (checked
//     at runtime - contiguity alone does not guarantee alignment, see the
//     launcher); given both, every row offset (row*N elements) is a multiple
//     of 16 bytes and the Vec casts below are valid. 8-wide fp16 loads
//     matter: with 4-wide (8-byte) loads the fp16 kernel measured well
//     behind PyTorch at large shapes, with 16-byte loads it is at parity or
//     ahead (interim development measurement; the committed data covers the
//     shipped 16-byte version).
//   * Single-pass Welford statistics instead of two passes: mean and the
//     centred sum of squares (m2) are maintained together while the row is
//     read ONCE, then partial (n, mean, m2) triples are merged across the
//     block with Chan's parallel update. This removes one full read of the
//     row; at memory-bound shapes (thousands of rows) that read is the
//     difference between ~0.5x and ~1x of PyTorch's kernel time.
//
// The normalise pass still re-reads the row (as does PyTorch's kernel); that
// second read mostly hits L1/L2 because the same block just read it.
// ---------------------------------------------------------------------------

namespace {

// Elements per 16-byte vector load for each dtype.
template <typename scalar_t>
constexpr int kVecWidth = 16 / sizeof(scalar_t);

template <typename scalar_t>
struct alignas(16) Vec {
  scalar_t v[kVecWidth<scalar_t>];
};

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

template <typename scalar_t, typename acc_t, bool kGelu>
__global__ void layernorm_vec_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const scalar_t* __restrict__ gamma,
                                     const scalar_t* __restrict__ beta,
                                     int64_t N,  // multiple of kVecWidth<scalar_t>
                                     acc_t eps,
                                     bool use_tanh) {
  using V = Vec<scalar_t>;
  constexpr int kW = kVecWidth<scalar_t>;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t nvec = N / kW;

  const V* X = reinterpret_cast<const V*>(input + row * N);
  V* Y = reinterpret_cast<V*>(output + row * N);
  const V* G = reinterpret_cast<const V*>(gamma);  // may be null
  const V* B = reinterpret_cast<const V*>(beta);   // may be null

  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;

  // Single pass: per-thread Welford over a strided slice of the row.
  Welford<acc_t> w;
  for (int64_t i = tid; i < nvec; i += stride) {
    const V x = X[i];
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      const acc_t xv = static_cast<acc_t>(x.v[k]);
      w.n += 1;
      const acc_t delta = xv - w.mean;
      w.mean += delta / w.n;
      w.m2 += delta * (xv - w.mean);
    }
  }
  w = blockReduceWelford<acc_t>(w);
  if (tid == 0) {
    s_mean = w.mean;
    s_rstd = rsqrt(w.m2 / static_cast<acc_t>(N) + eps);
  }
  __syncthreads();
  const acc_t mean = s_mean;
  const acc_t rstd = s_rstd;

  // Normalise pass (the row re-read is mostly an L1/L2 hit). gamma/beta are
  // loaded as whole vectors up front rather than re-indexed per element, so
  // their traffic is one 16-byte load each regardless of what the compiler
  // does about common-subexpression elimination.
  for (int64_t i = tid; i < nvec; i += stride) {
    const V x = X[i];
    V gv, bv;
    if (gamma) gv = G[i];
    if (beta) bv = B[i];
    V y;
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      acc_t v = (static_cast<acc_t>(x.v[k]) - mean) * rstd;
      if (gamma) v *= static_cast<acc_t>(gv.v[k]);
      if (beta) v += static_cast<acc_t>(bv.v[k]);
      if (kGelu) v = use_tanh ? gelu_tanh<acc_t>(v) : gelu_erf<acc_t>(v);
      y.v[k] = static_cast<scalar_t>(v);
    }
    Y[i] = y;
  }
}

}  // namespace

void layernorm_cuda_launch(const at::Tensor& input2d,
                           const at::Tensor& weight_or_undefined,
                           const at::Tensor& bias_or_undefined,
                           at::Tensor& output2d,
                           double eps,
                           bool gelu,
                           bool gelu_tanh) {
  const int64_t M = input2d.size(0);
  const int64_t N = input2d.size(1);
  if (M == 0 || N == 0) return;  // nothing to do; also avoids a zero-sized grid launch
  // gridDim.x is a 32-bit quantity; one block per row.
  TORCH_CHECK(M <= 0x7fffffffLL, "layernorm: too many rows for a single launch (", M, ")");

  // Make sure we launch on the device that owns the tensors and on the stream PyTorch is
  // currently using (so the op orders correctly with surrounding work, is safe on side
  // streams and can be captured in a CUDA graph).
  c10::cuda::CUDAGuard device_guard(input2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Kernel choice (measured on an A100-SXM4-40GB, see benchmarks/results/):
  //   * Many rows, vectorisable layout: the vectorised single-pass kernel with
  //     small (<= 256 thread) blocks. With thousands of one-block rows the
  //     grid alone fills every SM, and per-row efficiency (16-byte loads, one
  //     fewer pass) decides throughput.
  //   * Few rows: the scalar two-pass kernel with up to 1024 threads per row.
  //     With only M <= a-few-hundred blocks the GPU is latency-bound and wide
  //     blocks put more threads to work; the vectorised kernel measured slower
  //     here for every tested shape.
  //   * Everything else: the scalar kernel. "Vectorisable" means N is a
  //     multiple of the per-dtype 16-byte vector width (4 x fp32, 8 x
  //     fp16/bf16, 2 x fp64) AND every tensor's data pointer is 16-byte
  //     aligned. Contiguity alone does not guarantee alignment: a contiguous
  //     1-D slice like base[1:] keeps its storage offset, so its data_ptr can
  //     sit 4 bytes into an allocation; PyTorch's own vectorised kernel makes
  //     the same runtime alignment check.
  constexpr int64_t kVecMinRows = 256;
  const int64_t vw = 16 / input2d.element_size();  // kVecWidth of the dispatched dtype
  const auto aligned16 = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
  };
  const bool ptrs_ok =
      aligned16(input2d.data_ptr()) && aligned16(output2d.data_ptr()) &&
      (!weight_or_undefined.defined() || aligned16(weight_or_undefined.data_ptr())) &&
      (!bias_or_undefined.defined() || aligned16(bias_or_undefined.data_ptr()));
  bool vec = ptrs_ok && (N % vw == 0) && (N >= 128) && (M >= kVecMinRows);

  // Debug/benchmark override: FUSED_LAYERNORM_FORCE_KERNEL=scalar|vec pins the
  // kernel choice (used to produce the committed scalar-baseline results).
  // Forcing "vec" still requires the layout to be vectorisable at all.
  static const char* const force_kernel = std::getenv("FUSED_LAYERNORM_FORCE_KERNEL");
  if (force_kernel != nullptr) {
    if (std::strcmp(force_kernel, "scalar") == 0) vec = false;
    if (std::strcmp(force_kernel, "vec") == 0) vec = ptrs_ok && (N % vw == 0);
  }

  // Block size, always a power of two (keeps the reductions' warp arithmetic
  // exact). Scalar path: smallest power of two >= N, clamped to [32, 1024] (a
  // fixed 1024-thread block would leave most warps idle for short rows, e.g.
  // N = 64 would run 2 useful warps and 30 idle ones through every barrier).
  // Vectorised path: about two 16-byte vectors per thread, clamped to
  // [64, 256] — 128 threads for fp32 N = 1024 matches PyTorch's
  // vectorised-kernel block size, and larger rows saturate with 256 threads;
  // bigger blocks measured slower on narrow rows, smaller ones on wide rows.
  const int64_t items = vec ? (N / vw + 1) / 2 : N;
  const int floor_threads = vec ? 2 * kWarpSize : kWarpSize;
  const int cap = vec ? 256 : kMaxThreads;
  int threads = floor_threads;
  while (threads < items && threads < cap) threads *= 2;

  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input2d.scalar_type(), "layernorm_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* x = input2d.data_ptr<scalar_t>();
        scalar_t* y = output2d.data_ptr<scalar_t>();
        const scalar_t* g =
            weight_or_undefined.defined() ? weight_or_undefined.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_or_undefined.defined() ? bias_or_undefined.data_ptr<scalar_t>() : nullptr;
        const acc_t eps_acc = static_cast<acc_t>(eps);
        if (vec) {
          if (gelu) {
            layernorm_vec_kernel<scalar_t, acc_t, true>
                <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, gelu_tanh);
          } else {
            layernorm_vec_kernel<scalar_t, acc_t, false>
                <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, /*use_tanh=*/false);
          }
        } else if (gelu) {
          layernorm_kernel<scalar_t, acc_t, true>
              <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, gelu_tanh);
        } else {
          layernorm_kernel<scalar_t, acc_t, false>
              <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, /*use_tanh=*/false);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
