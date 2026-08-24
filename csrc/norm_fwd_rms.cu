// RMSNorm forward: instantiations of the generic kernels in
// norm_fwd_kernels.cuh for kRMS=true, and their host launcher.
//
//   y = x * rsqrt(mean(x^2) + eps) [* weight]        (plain)
//   z = round(x + residual); y = rms_norm(z)         (fused-add; residual_out
//                                                     receives z, may alias
//                                                     residual_in)
//
// Epilogues in this TU: kNone (all dtypes incl. fp64, which exists so
// torch.autograd.gradcheck can exercise the autograd path) and
// kFp8Static/kFp8Dynamic (fp32/fp16/bf16 only; output dtype float8_e4m3fn,
// inference-only - the RMS family is the one LLM serving stacks quantise).

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
#include "norm_epilogue.cuh"
#include "norm_fwd_kernels.cuh"
#include "norm_vec.cuh"

namespace {

using fused_norm::EpilogueParams;
using fused_norm::EpiNone;
using fused_norm::NormEpilogue;

template <typename scalar_t, typename acc_t, bool kFusedAdd, typename Epi>
void launch_rms_kernel(const at::Tensor& input2d,
                       const at::Tensor& residual_in2d,
                       const at::Tensor& weight_or_undef,
                       typename Epi::out_t* y,
                       at::Tensor& residual_out2d,
                       at::Tensor& rstd_or_undef,
                       double eps,
                       Epi epi,
                       bool vec,
                       dim3 grid,
                       dim3 block,
                       cudaStream_t stream) {
  const scalar_t* x = input2d.data_ptr<scalar_t>();
  const scalar_t* r = kFusedAdd ? residual_in2d.data_ptr<scalar_t>() : nullptr;
  scalar_t* z = kFusedAdd ? residual_out2d.data_ptr<scalar_t>() : nullptr;
  const scalar_t* g =
      weight_or_undef.defined() ? weight_or_undef.data_ptr<scalar_t>() : nullptr;
  acc_t* rstd = rstd_or_undef.defined() ? rstd_or_undef.data_ptr<acc_t>() : nullptr;
  const acc_t eps_acc = static_cast<acc_t>(eps);
  const int64_t N = input2d.size(1);

  if (vec) {
    fused_norm::norm_fwd_vec_kernel<scalar_t, acc_t, /*kRMS=*/true, kFusedAdd>
        <<<grid, block, 0, stream>>>(x, r, z, y, g, /*beta=*/nullptr,
                                     /*mean_out=*/nullptr, rstd, N, eps_acc, epi);
  } else {
    fused_norm::norm_fwd_kernel<scalar_t, acc_t, /*kRMS=*/true, kFusedAdd>
        <<<grid, block, 0, stream>>>(x, r, z, y, g, /*beta=*/nullptr,
                                     /*mean_out=*/nullptr, rstd, N, eps_acc, epi);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void rmsnorm_fwd_cuda_launch(const at::Tensor& input2d,
                             const at::Tensor& residual_in2d_or_undef,
                             const at::Tensor& weight_or_undef,
                             at::Tensor& output2d,
                             at::Tensor& residual_out2d_or_undef,
                             at::Tensor& rstd_or_undef,
                             double eps,
                             fused_norm::NormEpilogue epi,
                             const fused_norm::EpilogueParams& epi_params) {
  const int64_t M = input2d.size(0);
  const int64_t N = input2d.size(1);
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= 0x7fffffffLL, "rms_norm: too many rows for a single launch (", M, ")");
  TORCH_CHECK(epi == NormEpilogue::kNone || epi == NormEpilogue::kFp8Static ||
                  epi == NormEpilogue::kFp8Dynamic,
              "rmsnorm_fwd_cuda_launch: unsupported epilogue");
  const bool fused_add = residual_in2d_or_undef.defined();
  TORCH_CHECK(fused_add == residual_out2d_or_undef.defined(),
              "residual_in and residual_out must be given together");

  c10::cuda::CUDAGuard device_guard(input2d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t vw = 16 / input2d.element_size();
  bool vectorisable =
      (N % vw == 0) && fused_norm::aligned16(input2d.data_ptr()) &&
      fused_norm::aligned16(output2d.data_ptr()) &&
      (!weight_or_undef.defined() || fused_norm::aligned16(weight_or_undef.data_ptr()));
  if (fused_add) {
    vectorisable = vectorisable &&
                   fused_norm::aligned16(residual_in2d_or_undef.data_ptr()) &&
                   fused_norm::aligned16(residual_out2d_or_undef.data_ptr());
  }
  const bool vec = fused_norm::choose_vec(vectorisable, M, N);
  const int threads = fused_norm::choose_block_size(vec, N, vw);

  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input2d.scalar_type(), "rmsnorm_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

        auto dispatch_fused = [&](auto epi_functor, auto* y_ptr) {
          if (fused_add) {
            launch_rms_kernel<scalar_t, acc_t, /*kFusedAdd=*/true>(
                input2d, residual_in2d_or_undef, weight_or_undef, y_ptr,
                residual_out2d_or_undef, rstd_or_undef, eps, epi_functor, vec, grid, block,
                stream);
          } else {
            launch_rms_kernel<scalar_t, acc_t, /*kFusedAdd=*/false>(
                input2d, residual_in2d_or_undef, weight_or_undef, y_ptr,
                residual_out2d_or_undef, rstd_or_undef, eps, epi_functor, vec, grid, block,
                stream);
          }
        };

        if (epi == NormEpilogue::kNone) {
          dispatch_fused(EpiNone<scalar_t, acc_t>{}, output2d.data_ptr<scalar_t>());
          return;
        }

#if FUSED_NORM_HAS_FP8
        if constexpr (!std::is_same_v<scalar_t, double>) {
          TORCH_CHECK(output2d.scalar_type() == at::kFloat8_e4m3fn,
                      "fp8 epilogue needs a float8_e4m3fn output tensor");
          auto* y8 = reinterpret_cast<__nv_fp8_e4m3*>(output2d.data_ptr());
          if (epi == NormEpilogue::kFp8Static) {
            TORCH_CHECK(epi_params.scale_in.defined() &&
                            epi_params.scale_in.scalar_type() == at::kFloat &&
                            epi_params.scale_in.numel() == 1 && epi_params.scale_in.is_cuda(),
                        "fp8 static scale must be a 1-element fp32 CUDA tensor");
            fused_norm::EpiFp8Static<scalar_t, acc_t> e{
                epi_params.scale_in.data_ptr<float>()};
            dispatch_fused(e, y8);
          } else {
            TORCH_CHECK(epi_params.scale_out.defined() &&
                            epi_params.scale_out.scalar_type() == at::kFloat &&
                            epi_params.scale_out.numel() == M && epi_params.scale_out.is_cuda(),
                        "fp8 dynamic scale_out must be an [M] fp32 CUDA tensor");
            fused_norm::EpiFp8Dynamic<scalar_t, acc_t> e{
                epi_params.scale_out.data_ptr<float>(),
                static_cast<float>(epi_params.scale_ub)};
            dispatch_fused(e, y8);
          }
          return;
        }
#endif
        TORCH_CHECK(false, "fp8 epilogues support float32/float16/bfloat16 inputs only");
      });
}
