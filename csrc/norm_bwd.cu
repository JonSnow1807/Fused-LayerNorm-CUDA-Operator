// Backward kernels for LayerNorm and RMSNorm (plain and fused-add).
//
// Notation: xz is the tensor the forward normalised - the input for plain
// norms, the rounded sum z = round(x + residual) for fused-add (which the
// forward already returns, so saving it costs no extra memory). mean/rstd are
// the forward's saved per-row statistics (acc dtype). gamma is the forward
// weight (null => 1). All math in acc_t.
//
// dx (one block per row, scalar strided):
//   LayerNorm: xhat = (xz - mean) * rstd;  g = dy * gamma
//     dx = rstd * (g - sum(g)/N - xhat * sum(g*xhat)/N)        [+ dz_extra]
//   RMSNorm:   xhat = xz * rstd;           g = dy * gamma
//     dx = rstd * (g - xhat * sum(g*xhat)/N)                   [+ dz_extra]
//   The two row sums reduce together (Sum2). dz_extra is the downstream
//   cotangent of the fused-add op's second output (new_residual): because
//   z = x + residual, dx = dresidual = norm_dx + dz_extra, so the launcher
//   adds it elementwise here and the same tensor serves both input grads.
//
// dgamma/dbeta (two-stage, DETERMINISTIC - no atomics):
//   Stage 1 (here): grid (ceil(N/32), num_chunks), 32x32 blocks. Each block
//   owns a 32-column slice and a fixed chunk of rows; per-thread fp32
//   accumulation over its rows, shared-memory transpose reduce, one write of
//   partials[chunk, col]:  dgamma_p = sum(dy * xhat), dbeta_p = sum(dy).
//   Stage 2 (bindings): partials.sum(0).to(param dtype) - a fixed-shape aten
//   reduction, so parameter grads are bitwise run-to-run reproducible.
//   Atomics would be faster to write and nondeterministic; determinism wins.
//
// These kernels are correctness-first (scalar loads); backward bandwidth is
// benchmarked honestly, not assumed. fp32/fp16/bf16 + fp64 (for gradcheck).

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <type_traits>

#include "layernorm.h"
#include "norm_dispatch.cuh"
#include "norm_reduce.cuh"

namespace {

using fused_norm::blockReduceSum2;
using fused_norm::Sum2;

template <typename scalar_t, typename acc_t, bool kRMS>
__global__ void norm_bwd_dx_kernel(const scalar_t* __restrict__ dy,
                                   const scalar_t* __restrict__ dz_extra,  // null => 0
                                   const scalar_t* __restrict__ xz,
                                   const acc_t* __restrict__ mean,         // null iff kRMS
                                   const acc_t* __restrict__ rstd,
                                   const scalar_t* __restrict__ gamma,     // null => 1
                                   scalar_t* __restrict__ dx,
                                   int64_t N) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  const scalar_t* DY = dy + row * N;
  const scalar_t* XZ = xz + row * N;
  const scalar_t* DZ = dz_extra != nullptr ? dz_extra + row * N : nullptr;
  scalar_t* DX = dx + row * N;
  const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
  const acc_t rs = rstd[row];

  __shared__ acc_t s_c1;  // sum(g) / N          (LayerNorm only; 0 for RMS)
  __shared__ acc_t s_c2;  // sum(g * xhat) / N

  Sum2<acc_t> sums;
  for (int64_t i = tid; i < N; i += stride) {
    const acc_t g = static_cast<acc_t>(DY[i]) *
                    (gamma != nullptr ? static_cast<acc_t>(gamma[i]) : static_cast<acc_t>(1));
    const acc_t xhat = (static_cast<acc_t>(XZ[i]) - mu) * rs;
    if constexpr (!kRMS) sums.a += g;
    sums.b += g * xhat;
  }
  sums = blockReduceSum2<acc_t>(sums);
  if (tid == 0) {
    s_c1 = kRMS ? static_cast<acc_t>(0) : sums.a / static_cast<acc_t>(N);
    s_c2 = sums.b / static_cast<acc_t>(N);
  }
  __syncthreads();
  const acc_t c1 = s_c1;
  const acc_t c2 = s_c2;

  for (int64_t i = tid; i < N; i += stride) {
    const acc_t g = static_cast<acc_t>(DY[i]) *
                    (gamma != nullptr ? static_cast<acc_t>(gamma[i]) : static_cast<acc_t>(1));
    const acc_t xhat = (static_cast<acc_t>(XZ[i]) - mu) * rs;
    acc_t v = rs * (g - c1 - xhat * c2);
    if (DZ != nullptr) v += static_cast<acc_t>(DZ[i]);
    DX[i] = static_cast<scalar_t>(v);
  }
}

// Stage 1 of the parameter gradients. kTile = 32.
constexpr int kTile = 32;

template <typename scalar_t, typename acc_t, bool kRMS, bool kBeta>
__global__ void norm_bwd_param_partials_kernel(const scalar_t* __restrict__ dy,
                                               const scalar_t* __restrict__ xz,
                                               const acc_t* __restrict__ mean,  // null iff kRMS
                                               const acc_t* __restrict__ rstd,
                                               acc_t* __restrict__ dgamma_partials,  // [chunks, N]
                                               acc_t* __restrict__ dbeta_partials,   // [chunks, N] or null
                                               int64_t M,
                                               int64_t N,
                                               int64_t rows_per_chunk) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x;
  const int64_t chunk = blockIdx.y;
  const int64_t row_begin = chunk * rows_per_chunk;
  const int64_t row_end = row_begin + rows_per_chunk < M ? row_begin + rows_per_chunk : M;

  acc_t dg = 0;
  acc_t db = 0;
  if (col < N) {
    for (int64_t row = row_begin + threadIdx.y; row < row_end; row += kTile) {
      const acc_t d = static_cast<acc_t>(dy[row * N + col]);
      const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
      const acc_t xhat = (static_cast<acc_t>(xz[row * N + col]) - mu) * rstd[row];
      dg += d * xhat;
      if constexpr (kBeta) db += d;
    }
  }

  // Transpose-free tree reduce over threadIdx.y (the row dimension of the
  // block): +33 padding keeps the column accesses bank-conflict-free.
  __shared__ acc_t s_dg[kTile][kTile + 1];
  __shared__ acc_t s_db[kTile][kTile + 1];
  s_dg[threadIdx.y][threadIdx.x] = dg;
  if constexpr (kBeta) s_db[threadIdx.y][threadIdx.x] = db;
  __syncthreads();
  for (int offset = kTile / 2; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      s_dg[threadIdx.y][threadIdx.x] += s_dg[threadIdx.y + offset][threadIdx.x];
      if constexpr (kBeta) s_db[threadIdx.y][threadIdx.x] += s_db[threadIdx.y + offset][threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0 && col < N) {
    dgamma_partials[chunk * N + col] = s_dg[0][threadIdx.x];
    if constexpr (kBeta) dbeta_partials[chunk * N + col] = s_db[0][threadIdx.x];
  }
}

}  // namespace

void norm_bwd_dx_cuda_launch(bool rms,
                             const at::Tensor& dy2d,
                             const at::Tensor& dz_extra2d_or_undef,
                             const at::Tensor& xz2d,
                             const at::Tensor& mean_or_undef,
                             const at::Tensor& rstd,
                             const at::Tensor& weight_or_undef,
                             at::Tensor& dx2d) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= 0x7fffffffLL, "norm backward: too many rows (", M, ")");
  TORCH_CHECK(rms == !mean_or_undef.defined(), "mean must be given iff LayerNorm");

  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = fused_norm::choose_block_size(/*vec=*/false, N, /*vec_width=*/1);
  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xz2d.scalar_type(), "norm_bwd_dx", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* dyp = dy2d.data_ptr<scalar_t>();
        const scalar_t* dzp =
            dz_extra2d_or_undef.defined() ? dz_extra2d_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* xzp = xz2d.data_ptr<scalar_t>();
        const acc_t* meanp = mean_or_undef.defined() ? mean_or_undef.data_ptr<acc_t>() : nullptr;
        const acc_t* rstdp = rstd.data_ptr<acc_t>();
        const scalar_t* g =
            weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
        scalar_t* dxp = dx2d.data_ptr<scalar_t>();
        if (rms) {
          norm_bwd_dx_kernel<scalar_t, acc_t, /*kRMS=*/true>
              <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g, dxp, N);
        } else {
          norm_bwd_dx_kernel<scalar_t, acc_t, /*kRMS=*/false>
              <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g, dxp, N);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void norm_bwd_param_partials_cuda_launch(bool rms,
                                         const at::Tensor& dy2d,
                                         const at::Tensor& xz2d,
                                         const at::Tensor& mean_or_undef,
                                         const at::Tensor& rstd,
                                         at::Tensor& dgamma_partials,
                                         at::Tensor& dbeta_partials_or_undef) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(rms == !mean_or_undef.defined(), "mean must be given iff LayerNorm");
  const int64_t num_chunks = dgamma_partials.size(0);
  const int64_t rows_per_chunk = (M + num_chunks - 1) / num_chunks;
  const bool has_beta = dbeta_partials_or_undef.defined();

  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>((N + kTile - 1) / kTile),
                  static_cast<unsigned int>(num_chunks));
  const dim3 block(kTile, kTile);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xz2d.scalar_type(), "norm_bwd_params", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* dyp = dy2d.data_ptr<scalar_t>();
        const scalar_t* xzp = xz2d.data_ptr<scalar_t>();
        const acc_t* meanp = mean_or_undef.defined() ? mean_or_undef.data_ptr<acc_t>() : nullptr;
        const acc_t* rstdp = rstd.data_ptr<acc_t>();
        acc_t* dgp = dgamma_partials.data_ptr<acc_t>();
        acc_t* dbp =
            has_beta ? dbeta_partials_or_undef.data_ptr<acc_t>() : nullptr;
        auto launch = [&](auto rms_tag, auto beta_tag) {
          norm_bwd_param_partials_kernel<scalar_t, acc_t, decltype(rms_tag)::value,
                                         decltype(beta_tag)::value>
              <<<grid, block, 0, stream>>>(dyp, xzp, meanp, rstdp, dgp, dbp, M, N,
                                           rows_per_chunk);
        };
        if (rms) {
          launch(std::true_type{}, std::false_type{});
        } else if (has_beta) {
          launch(std::false_type{}, std::true_type{});
        } else {
          launch(std::false_type{}, std::false_type{});
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
