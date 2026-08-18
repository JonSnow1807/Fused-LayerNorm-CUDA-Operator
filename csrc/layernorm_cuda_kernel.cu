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

  // Block size: smallest power of two in [32, 1024] that is >= N. A fixed 1024-thread block
  // would leave most warps idle for short rows (e.g. N = 64 would run 2 useful warps and 30
  // idle ones through every barrier); rows longer than 1024 are handled by the strided
  // loops in the kernel. Power of two keeps blockReduceSum's warp arithmetic exact.
  int threads = kWarpSize;
  while (threads < N && threads < kMaxThreads) threads *= 2;

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
        if (gelu) {
          layernorm_kernel<scalar_t, acc_t, true>
              <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, gelu_tanh);
        } else {
          layernorm_kernel<scalar_t, acc_t, false>
              <<<grid, block, 0, stream>>>(x, y, g, b, N, eps_acc, /*use_tanh=*/false);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
