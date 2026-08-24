// Fused LayerNorm (+ optional GELU) forward kernels for PyTorch.
//
// Design (deliberately simple; readability is the point of this repository):
//   * One thread block per row. The row of length N is normalised over its last dimension.
//   * Two kernels: a scalar two-pass kernel (mean, then centred sum of squares) and a
//     16-byte-vectorised single-pass Welford kernel; the launcher picks per call using the
//     measured policy in norm_dispatch.cuh.
//   * All arithmetic is done in acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>
//     (float for float/half/bfloat16, double for double); only the final store casts back.
//   * These kernels are forward only; the v0.4.0 backward kernels live in their own TUs.
//
// The shared device machinery (reductions, Welford, Vec, GELU) was extracted verbatim into
// the norm_*.cuh headers when v0.4.0 added more ops; the two kernels below are the ones
// whose numerics the committed A100 runs cover, and they are unchanged.
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
#include "norm_dispatch.cuh"
#include "norm_epilogue.cuh"
#include "norm_reduce.cuh"
#include "norm_vec.cuh"

namespace {

using fused_norm::blockReduceSum;
using fused_norm::blockReduceWelford;
using fused_norm::gelu_erf;
using fused_norm::gelu_tanh;
using fused_norm::kVecWidth;
using fused_norm::NormEpilogue;
using fused_norm::Vec;
using fused_norm::Welford;

// The scalar two-pass kernel. One block handles one row of N elements.
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

// ---------------------------------------------------------------------------
// Vectorised single-pass variant (used when the layout is vectorisable,
// N >= 128 and there are >= 256 rows; see norm_dispatch.cuh).
//
// Two changes relative to layernorm_kernel above, both copied from what
// PyTorch's own vectorized_layer_norm_kernel does:
//   * 16-byte aligned vector loads/stores (kVecWidth elements: 4 x float,
//     8 x half/bfloat16, 2 x double). The launcher takes this path only when
//     N % kVecWidth == 0 AND every data pointer is 16-byte aligned (checked
//     at runtime - see aligned16 in norm_vec.cuh); given both, every row
//     offset (row*N elements) is a multiple of 16 bytes and the Vec casts
//     below are valid. 8-wide fp16 loads matter: with 4-wide (8-byte) loads
//     the fp16 kernel measured well behind PyTorch at large shapes, with
//     16-byte loads it is at parity or ahead (interim development
//     measurement; the committed data covers the shipped 16-byte version).
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
                           NormEpilogue epi) {
  const int64_t M = input2d.size(0);
  const int64_t N = input2d.size(1);
  if (M == 0 || N == 0) return;  // nothing to do; also avoids a zero-sized grid launch
  // gridDim.x is a 32-bit quantity; one block per row.
  TORCH_CHECK(M <= 0x7fffffffLL, "layernorm: too many rows for a single launch (", M, ")");
  TORCH_CHECK(epi == NormEpilogue::kNone || epi == NormEpilogue::kGeluErf ||
                  epi == NormEpilogue::kGeluTanh,
              "layernorm_cuda_launch supports only the none/gelu epilogues");

  // Make sure we launch on the device that owns the tensors and on the stream PyTorch is
  // currently using (so the op orders correctly with surrounding work, is safe on side
  // streams and can be captured in a CUDA graph).
  c10::cuda::CUDAGuard device_guard(input2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t vw = 16 / input2d.element_size();  // kVecWidth of the dispatched dtype
  const bool vectorisable =
      (N % vw == 0) && fused_norm::aligned16(input2d.data_ptr()) &&
      fused_norm::aligned16(output2d.data_ptr()) &&
      (!weight_or_undefined.defined() || fused_norm::aligned16(weight_or_undefined.data_ptr())) &&
      (!bias_or_undefined.defined() || fused_norm::aligned16(bias_or_undefined.data_ptr()));
  const bool vec = fused_norm::choose_vec(vectorisable, M, N);
  const int threads = fused_norm::choose_block_size(vec, N, vw);

  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  const bool gelu = epi != NormEpilogue::kNone;
  const bool gelu_tanh = epi == NormEpilogue::kGeluTanh;

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
