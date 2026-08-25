// Backward launchers for LayerNorm and RMSNorm (plain and fused-add).
// The kernel templates live in norm_bwd_kernels.cuh (math and determinism
// contract documented there). fp32/fp16/bf16 + fp64 (for gradcheck).

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

using fused_norm::NormEpilogue;

void norm_bwd_dx_cuda_launch(bool rms,
                             const at::Tensor& dy2d,
                             const at::Tensor& dz_extra2d_or_undef,
                             const at::Tensor& xz2d,
                             const at::Tensor& mean_or_undef,
                             const at::Tensor& rstd,
                             const at::Tensor& weight_or_undef,
                             const at::Tensor& bias_or_undef,
                             at::Tensor& dx2d,
                             NormEpilogue act) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= 0x7fffffffLL, "norm backward: too many rows (", M, ")");
  TORCH_CHECK(rms == !mean_or_undef.defined(), "mean must be given iff LayerNorm");
  if (act != NormEpilogue::kNone) {
    TORCH_CHECK(!rms, "fused activation backward is LayerNorm-only");
    TORCH_CHECK(act == NormEpilogue::kGeluErf || act == NormEpilogue::kGeluTanh,
                "unsupported activation in norm backward");
    norm_bwd_dx_gelu_cuda(act == NormEpilogue::kGeluTanh, dy2d, dz_extra2d_or_undef, xz2d,
                          mean_or_undef, rstd, weight_or_undef, bias_or_undef, dx2d);
    return;
  }

  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>(M));

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

        // Same vec policy as the forward: every participating pointer 16-byte
        // aligned and N a multiple of the dtype's vector width. mean/rstd are
        // per-row acc_t scalars — no gate needed for them.
        constexpr int kW = fused_norm::kVecWidth<scalar_t>;
        const bool vectorisable = (N % kW == 0) && fused_norm::aligned16(dyp) &&
                                  fused_norm::aligned16(xzp) && fused_norm::aligned16(dxp) &&
                                  (g == nullptr || fused_norm::aligned16(g)) &&
                                  (dzp == nullptr || fused_norm::aligned16(dzp));
        const bool vec = fused_norm::choose_vec(vectorisable, M, N);
        const dim3 block(static_cast<unsigned int>(fused_norm::choose_block_size(vec, N, kW)));

        auto launch = [&](auto rms_tag) {
          constexpr bool kRMS = decltype(rms_tag)::value;
          if (vec) {
            fused_norm::norm_bwd_dx_vec_kernel<scalar_t, acc_t, kRMS>
                <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g,
                                             /*beta=*/nullptr, dxp, N);
          } else {
            fused_norm::norm_bwd_dx_kernel<scalar_t, acc_t, kRMS>
                <<<grid, block, 0, stream>>>(dyp, dzp, xzp, meanp, rstdp, g,
                                             /*beta=*/nullptr, dxp, N);
          }
        };
        if (rms) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void norm_bwd_param_partials_cuda_launch(bool rms,
                                         const at::Tensor& dy2d,
                                         const at::Tensor& xz2d,
                                         const at::Tensor& mean_or_undef,
                                         const at::Tensor& rstd,
                                         const at::Tensor& weight_or_undef,
                                         const at::Tensor& bias_or_undef,
                                         at::Tensor& dgamma_partials,
                                         at::Tensor& dbeta_partials_or_undef,
                                         NormEpilogue act) {
  const int64_t M = xz2d.size(0);
  const int64_t N = xz2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(rms == !mean_or_undef.defined(), "mean must be given iff LayerNorm");
  if (act != NormEpilogue::kNone) {
    TORCH_CHECK(!rms, "fused activation backward is LayerNorm-only");
    TORCH_CHECK(act == NormEpilogue::kGeluErf || act == NormEpilogue::kGeluTanh,
                "unsupported activation in norm backward");
    norm_bwd_param_partials_gelu_cuda(act == NormEpilogue::kGeluTanh, dy2d, xz2d,
                                      mean_or_undef, rstd, weight_or_undef, bias_or_undef,
                                      dgamma_partials, dbeta_partials_or_undef);
    return;
  }
  const int64_t num_chunks = dgamma_partials.size(0);
  const int64_t rows_per_chunk = (M + num_chunks - 1) / num_chunks;
  const bool has_beta = dbeta_partials_or_undef.defined();

  c10::cuda::CUDAGuard device_guard(xz2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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

        // Own eligibility rule (this is a column-tiled grid, so choose_vec's
        // M-floor for block-per-row grids does not apply): vectorisable
        // layout and at least one full block width of columns.
        constexpr int kW = fused_norm::kVecWidth<scalar_t>;
        const bool vectorisable =
            (N % kW == 0) && fused_norm::aligned16(dyp) && fused_norm::aligned16(xzp);
        const bool vec = fused_norm::apply_force_kernel_env(
            vectorisable && N >= fused_norm::kPTx * kW, vectorisable);

        auto launch = [&](auto rms_tag, auto beta_tag) {
          constexpr bool kRMS = decltype(rms_tag)::value;
          constexpr bool kBeta = decltype(beta_tag)::value;
          if (vec) {
            const int64_t nvec = N / kW;
            const dim3 grid(
                static_cast<unsigned int>((nvec + fused_norm::kPTx - 1) / fused_norm::kPTx),
                static_cast<unsigned int>(num_chunks));
            const dim3 block(fused_norm::kPTx, fused_norm::kPTy);
            fused_norm::norm_bwd_param_partials_vec_kernel<scalar_t, acc_t, kRMS, kBeta>
                <<<grid, block, 0, stream>>>(dyp, xzp, meanp, rstdp, /*gamma=*/nullptr,
                                             /*beta=*/nullptr, dgp, dbp, M, N,
                                             rows_per_chunk);
          } else {
            constexpr int kTile = fused_norm::kBwdTile;
            const dim3 grid(static_cast<unsigned int>((N + kTile - 1) / kTile),
                            static_cast<unsigned int>(num_chunks));
            const dim3 block(kTile, kTile);
            fused_norm::norm_bwd_param_partials_kernel<scalar_t, acc_t, kRMS, kBeta>
                <<<grid, block, 0, stream>>>(dyp, xzp, meanp, rstdp, /*gamma=*/nullptr,
                                             /*beta=*/nullptr, dgp, dbp, M, N,
                                             rows_per_chunk);
          }
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

void norm_bwd_param_finalize_cuda_launch(const at::Tensor& dgamma_partials,
                                         const at::Tensor& dbeta_partials_or_undef,
                                         at::Tensor& dgamma_or_undef,
                                         at::Tensor& dbeta_or_undef) {
  const int64_t chunks = dgamma_partials.size(0);
  const int64_t N = dgamma_partials.size(1);
  if (N == 0) return;
  TORCH_CHECK(dgamma_or_undef.defined() || dbeta_or_undef.defined(),
              "finalize called with no requested output");
  const at::Tensor& out_for_dtype =
      dgamma_or_undef.defined() ? dgamma_or_undef : dbeta_or_undef;

  c10::cuda::CUDAGuard device_guard(dgamma_partials.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<unsigned int>((N + fused_norm::kFTx - 1) / fused_norm::kFTx));
  const dim3 blockdim(fused_norm::kFTx, fused_norm::kFTy);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, out_for_dtype.scalar_type(),
      "norm_bwd_param_finalize", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const acc_t* dgp = dgamma_partials.data_ptr<acc_t>();
        const acc_t* dbp = dbeta_partials_or_undef.defined()
                               ? dbeta_partials_or_undef.data_ptr<acc_t>()
                               : nullptr;
        scalar_t* dg_out =
            dgamma_or_undef.defined() ? dgamma_or_undef.data_ptr<scalar_t>() : nullptr;
        scalar_t* db_out =
            dbeta_or_undef.defined() ? dbeta_or_undef.data_ptr<scalar_t>() : nullptr;
        fused_norm::norm_bwd_param_finalize_kernel<scalar_t, acc_t>
            <<<grid, blockdim, 0, stream>>>(dgp, dbp, dg_out, db_out, chunks, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
