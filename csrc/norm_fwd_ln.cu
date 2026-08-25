// Fused residual-add + LayerNorm forward: instantiations of the generic
// kernels in norm_fwd_kernels.cuh for kRMS=false, kFusedAdd=true, and their
// host launcher.
//
//   z = round(x + residual)   (written to residual_out, which may alias
//                              residual_in for the in-place variant)
//   y = (z - mean(z)) * rsqrt(var(z) + eps) [* weight] [+ bias]
//
// Statistics are computed over the ROUNDED z, so y equals a plain LayerNorm
// of residual_out bitwise (composite equivalence). Plain (non-fused)
// LayerNorm keeps running on the original kernels in
// layernorm_cuda_kernel.cu; this TU deliberately instantiates only the
// fused-add combination to hold compile time down.
//
// Dtypes: fp32/fp16/bf16 and fp64 (for gradcheck). Epilogue: kNone only.

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "layernorm.h"
#include "norm_dispatch.cuh"
#include "norm_epilogue.cuh"
#include "norm_fwd_kernels.cuh"
#include "norm_vec.cuh"

namespace {

using fused_norm::EpiGeluErf;
using fused_norm::EpiGeluTanh;
using fused_norm::EpiNone;
using fused_norm::NormEpilogue;

}  // namespace

void layernorm_fwd_train_cuda_launch(const at::Tensor& input2d,
                                     const at::Tensor& weight_or_undef,
                                     const at::Tensor& bias_or_undef,
                                     at::Tensor& output2d,
                                     at::Tensor& mean2d,
                                     at::Tensor& rstd2d,
                                     double eps,
                                     fused_norm::NormEpilogue epi_kind) {
  TORCH_CHECK(epi_kind == NormEpilogue::kNone || epi_kind == NormEpilogue::kGeluErf ||
                  epi_kind == NormEpilogue::kGeluTanh,
              "layer_norm_fwd_train: unsupported epilogue");
  const int64_t M = input2d.size(0);
  const int64_t N = input2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= 0x7fffffffLL,
              "layer_norm_fwd_train: too many rows for a single launch (", M, ")");

  c10::cuda::CUDAGuard device_guard(input2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t vw = 16 / input2d.element_size();
  const bool vectorisable =
      (N % vw == 0) && fused_norm::aligned16(input2d.data_ptr()) &&
      fused_norm::aligned16(output2d.data_ptr()) &&
      (!weight_or_undef.defined() || fused_norm::aligned16(weight_or_undef.data_ptr())) &&
      (!bias_or_undef.defined() || fused_norm::aligned16(bias_or_undef.data_ptr()));
  const bool vec = fused_norm::choose_vec(vectorisable, M, N);
  const int threads = fused_norm::choose_block_size(vec, N, vw);

  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input2d.scalar_type(),
      "layernorm_fwd_train_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* x = input2d.data_ptr<scalar_t>();
        scalar_t* y = output2d.data_ptr<scalar_t>();
        const scalar_t* g =
            weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_or_undef.defined() ? bias_or_undef.data_ptr<scalar_t>() : nullptr;
        acc_t* mean = mean2d.data_ptr<acc_t>();
        acc_t* rstd = rstd2d.data_ptr<acc_t>();
        const acc_t eps_acc = static_cast<acc_t>(eps);

        auto launch = [&](auto epi) {
          if (vec) {
            fused_norm::norm_fwd_vec_kernel<scalar_t, acc_t, /*kRMS=*/false,
                                            /*kFusedAdd=*/false>
                <<<grid, block, 0, stream>>>(x, nullptr, nullptr, y, g, b, mean, rstd, N,
                                             eps_acc, epi);
          } else {
            fused_norm::norm_fwd_kernel<scalar_t, acc_t, /*kRMS=*/false, /*kFusedAdd=*/false>
                <<<grid, block, 0, stream>>>(x, nullptr, nullptr, y, g, b, mean, rstd, N,
                                             eps_acc, epi);
          }
        };
        if (epi_kind == NormEpilogue::kGeluErf) {
          launch(EpiGeluErf<scalar_t, acc_t>{});
        } else if (epi_kind == NormEpilogue::kGeluTanh) {
          launch(EpiGeluTanh<scalar_t, acc_t>{});
        } else {
          launch(EpiNone<scalar_t, acc_t>{});
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void fused_add_layernorm_fwd_cuda_launch(const at::Tensor& input2d,
                                         const at::Tensor& residual_in2d,
                                         const at::Tensor& weight_or_undef,
                                         const at::Tensor& bias_or_undef,
                                         at::Tensor& output2d,
                                         at::Tensor& residual_out2d,
                                         at::Tensor& mean_or_undef,
                                         at::Tensor& rstd_or_undef,
                                         double eps) {
  const int64_t M = input2d.size(0);
  const int64_t N = input2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= 0x7fffffffLL,
              "fused_add_layer_norm: too many rows for a single launch (", M, ")");

  c10::cuda::CUDAGuard device_guard(input2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t vw = 16 / input2d.element_size();
  const bool vectorisable =
      (N % vw == 0) && fused_norm::aligned16(input2d.data_ptr()) &&
      fused_norm::aligned16(output2d.data_ptr()) &&
      fused_norm::aligned16(residual_in2d.data_ptr()) &&
      fused_norm::aligned16(residual_out2d.data_ptr()) &&
      (!weight_or_undef.defined() || fused_norm::aligned16(weight_or_undef.data_ptr())) &&
      (!bias_or_undef.defined() || fused_norm::aligned16(bias_or_undef.data_ptr()));
  const bool vec = fused_norm::choose_vec(vectorisable, M, N);
  const int threads = fused_norm::choose_block_size(vec, N, vw);

  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input2d.scalar_type(),
      "fused_add_layernorm_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const scalar_t* x = input2d.data_ptr<scalar_t>();
        const scalar_t* r = residual_in2d.data_ptr<scalar_t>();
        scalar_t* z = residual_out2d.data_ptr<scalar_t>();
        scalar_t* y = output2d.data_ptr<scalar_t>();
        const scalar_t* g =
            weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_or_undef.defined() ? bias_or_undef.data_ptr<scalar_t>() : nullptr;
        acc_t* mean = mean_or_undef.defined() ? mean_or_undef.data_ptr<acc_t>() : nullptr;
        acc_t* rstd = rstd_or_undef.defined() ? rstd_or_undef.data_ptr<acc_t>() : nullptr;
        const acc_t eps_acc = static_cast<acc_t>(eps);
        const EpiNone<scalar_t, acc_t> epi{};

        if (vec) {
          fused_norm::norm_fwd_vec_kernel<scalar_t, acc_t, /*kRMS=*/false, /*kFusedAdd=*/true>
              <<<grid, block, 0, stream>>>(x, r, z, y, g, b, mean, rstd, N, eps_acc, epi);
        } else {
          fused_norm::norm_fwd_kernel<scalar_t, acc_t, /*kRMS=*/false, /*kFusedAdd=*/true>
              <<<grid, block, 0, stream>>>(x, r, z, y, g, b, mean, rstd, N, eps_acc, epi);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
