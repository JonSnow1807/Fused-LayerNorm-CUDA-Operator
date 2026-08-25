// GELU legs of the backward launchers: the DGradGeluErf/DGradGeluTanh
// instantiations of the dx and parameter-partials kernels, isolated in their
// own TU so the erf/tanh-heavy kernels stay out of the hot-path norm_bwd.cu
// build (the same per-family split rationale as the forward TUs).
//
// LayerNorm family only (layer_norm_gelu is the only fused-activation op):
// y = gelu(h), h = xhat*gamma + beta, so the backward is the plain LayerNorm
// backward with dy replaced per element by dh = dy * gelu'(h), h recomputed
// in-kernel from xz/mean/rstd/gamma/beta — no extra M x N tensor is saved.
// fp32/fp16/bf16 + fp64 (for gradcheck).

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <type_traits>

#include "layernorm.h"
#include "norm_bwd_kernels.cuh"
#include "norm_dispatch.cuh"

void norm_bwd_dx_gelu_cuda(bool tanh_approx,
                           const at::Tensor& dy2d,
                           const at::Tensor& dz_extra2d_or_undef,
                           const at::Tensor& xz2d,
                           const at::Tensor& mean2d,
                           const at::Tensor& rstd,
                           const at::Tensor& weight_or_undef,
                           const at::Tensor& bias_or_undef,
                           at::Tensor& dx2d) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  const dim3 grid(static_cast<unsigned int>(M));
  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xz2d.scalar_type(), "norm_bwd_dx_gelu",
      [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* dyp = dy2d.data_ptr<scalar_t>();
        const scalar_t* dzp =
            dz_extra2d_or_undef.defined() ? dz_extra2d_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* xzp = xz2d.data_ptr<scalar_t>();
        const acc_t* meanp = mean2d.data_ptr<acc_t>();
        const acc_t* rstdp = rstd.data_ptr<acc_t>();
        const scalar_t* g =
            weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_or_undef.defined() ? bias_or_undef.data_ptr<scalar_t>() : nullptr;
        scalar_t* dxp = dx2d.data_ptr<scalar_t>();

        constexpr int kW = fused_norm::kVecWidth<scalar_t>;
        const bool vectorisable = (N % kW == 0) && fused_norm::aligned16(dyp) &&
                                  fused_norm::aligned16(xzp) && fused_norm::aligned16(dxp) &&
                                  (g == nullptr || fused_norm::aligned16(g)) &&
                                  (b == nullptr || fused_norm::aligned16(b)) &&
                                  (dzp == nullptr || fused_norm::aligned16(dzp));
        const bool vec = fused_norm::choose_vec(vectorisable, M, N);
        const dim3 block(static_cast<unsigned int>(fused_norm::choose_block_size(vec, N, kW)));

        auto launch = [&](auto dgrad) {
          using DG = decltype(dgrad);
          if (vec) {
            fused_norm::norm_bwd_dx_vec_kernel<scalar_t, acc_t, /*kRMS=*/false, DG>
                <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g, b, dxp, N, dgrad);
          } else {
            fused_norm::norm_bwd_dx_kernel<scalar_t, acc_t, /*kRMS=*/false, DG>
                <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g, b, dxp, N, dgrad);
          }
        };
        if (tanh_approx) {
          launch(fused_norm::DGradGeluTanh{});
        } else {
          launch(fused_norm::DGradGeluErf{});
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void norm_bwd_param_partials_gelu_cuda(bool tanh_approx,
                                       const at::Tensor& dy2d,
                                       const at::Tensor& xz2d,
                                       const at::Tensor& mean2d,
                                       const at::Tensor& rstd,
                                       const at::Tensor& weight_or_undef,
                                       const at::Tensor& bias_or_undef,
                                       at::Tensor& dgamma_partials,
                                       at::Tensor& dbeta_partials_or_undef) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  const int64_t num_chunks = dgamma_partials.size(0);
  const int64_t rows_per_chunk = (M + num_chunks - 1) / num_chunks;
  const bool has_beta = dbeta_partials_or_undef.defined();
  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xz2d.scalar_type(),
      "norm_bwd_params_gelu", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* dyp = dy2d.data_ptr<scalar_t>();
        const scalar_t* xzp = xz2d.data_ptr<scalar_t>();
        const acc_t* meanp = mean2d.data_ptr<acc_t>();
        const acc_t* rstdp = rstd.data_ptr<acc_t>();
        const scalar_t* g =
            weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_or_undef.defined() ? bias_or_undef.data_ptr<scalar_t>() : nullptr;
        acc_t* dgp = dgamma_partials.data_ptr<acc_t>();
        acc_t* dbp = has_beta ? dbeta_partials_or_undef.data_ptr<acc_t>() : nullptr;

        constexpr int kW = fused_norm::kVecWidth<scalar_t>;
        const bool vectorisable = (N % kW == 0) && fused_norm::aligned16(dyp) &&
                                  fused_norm::aligned16(xzp) &&
                                  (g == nullptr || fused_norm::aligned16(g)) &&
                                  (b == nullptr || fused_norm::aligned16(b));
        const bool vec = fused_norm::apply_force_kernel_env(
            vectorisable && N >= fused_norm::kPTx * kW, vectorisable);

        auto launch = [&](auto beta_tag, auto dgrad) {
          constexpr bool kBeta = decltype(beta_tag)::value;
          using DG = decltype(dgrad);
          if (vec) {
            const int64_t nvec = N / kW;
            const dim3 grid(
                static_cast<unsigned int>((nvec + fused_norm::kPTx - 1) / fused_norm::kPTx),
                static_cast<unsigned int>(num_chunks));
            const dim3 block(fused_norm::kPTx, fused_norm::kPTy);
            fused_norm::norm_bwd_param_partials_vec_kernel<scalar_t, acc_t, /*kRMS=*/false,
                                                           kBeta, DG>
                <<<grid, block, 0, stream>>>(dyp, xzp, meanp, rstdp, g, b, dgp, dbp, M, N,
                                             rows_per_chunk, dgrad);
          } else {
            constexpr int kTile = fused_norm::kBwdTile;
            const dim3 grid(static_cast<unsigned int>((N + kTile - 1) / kTile),
                            static_cast<unsigned int>(num_chunks));
            const dim3 block(kTile, kTile);
            fused_norm::norm_bwd_param_partials_kernel<scalar_t, acc_t, /*kRMS=*/false, kBeta,
                                                       DG>
                <<<grid, block, 0, stream>>>(dyp, xzp, meanp, rstdp, g, b, dgp, dbp, M, N,
                                             rows_per_chunk, dgrad);
          }
        };
        auto launch_beta = [&](auto dgrad) {
          if (has_beta) {
            launch(std::true_type{}, dgrad);
          } else {
            launch(std::false_type{}, dgrad);
          }
        };
        if (tanh_approx) {
          launch_beta(fused_norm::DGradGeluTanh{});
        } else {
          launch_beta(fused_norm::DGradGeluErf{});
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
