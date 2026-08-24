// Shared declarations between the translation units of the extension:
//   csrc/bindings.cpp             -- Python bindings, argument validation, reshape to/from 2-D
//   csrc/layernorm_cuda_kernel.cu -- the original LayerNorm(+GELU) kernels and their launcher
// Further TUs (RMSNorm, fused-add, backward) are added by v0.4.0 and declare
// their launchers here as they land.
#pragma once

#include <ATen/ATen.h>

namespace fused_norm {

// Which per-element epilogue the forward kernel applies after normalise+affine.
// An enum (not a growing list of bools) so launchers can dispatch through one
// switch and unsupported combinations are a TORCH_CHECK, not an accidental
// template instantiation.
enum class NormEpilogue : int {
  kNone = 0,
  kGeluErf = 1,
  kGeluTanh = 2,
  // kFp8Static / kFp8Dynamic are introduced with the quant kernels.
};

}  // namespace fused_norm

// Launches the fused LayerNorm(+GELU) kernel on the current CUDA stream.
//
// Preconditions (checked by the caller in bindings.cpp, NOT here):
//   * input2d  : contiguous CUDA tensor of shape (M, N), dtype float32/float64/float16/bfloat16
//   * output2d : contiguous CUDA tensor with the same shape/dtype/device as input2d
//   * weight_or_undefined / bias_or_undefined : either an undefined at::Tensor (affine term
//     omitted) or a contiguous 1-D CUDA tensor of length N with input2d's dtype/device.
//     They are independent: weight-only and bias-only are both valid.
//   * eps      : added to the biased variance before rsqrt
//   * epi      : kNone, kGeluErf or kGeluTanh (this launcher's kernels have no
//     other epilogues)
//
// M == 0 or N == 0 is a no-op (nothing is launched).
void layernorm_cuda_launch(const at::Tensor& input2d,
                           const at::Tensor& weight_or_undefined,
                           const at::Tensor& bias_or_undefined,
                           at::Tensor& output2d,
                           double eps,
                           fused_norm::NormEpilogue epi);

// Launches the RMSNorm forward kernel (norm_fwd_rms.cu) on the current CUDA
// stream. Same 2-D preconditions as above, plus:
//   * residual_in2d/residual_out2d: BOTH undefined (plain RMSNorm) or BOTH
//     defined contiguous (M, N) tensors of input2d's dtype/device (fused
//     residual-add: residual_out receives round(input + residual_in), and the
//     norm consumes that rounded sum). residual_out MAY alias residual_in
//     (in-place).
//   * There is no bias (RMSNorm has none).
//   * rstd_or_undef: undefined, or a contiguous (M,) tensor of the
//     accumulation dtype (fp32; fp64 for double inputs) that receives the
//     per-row rsqrt(mean(z^2) + eps) - what autograd saves.
//   * eps must already be resolved (the F.rms_norm eps=None convention is a
//     Python-side concern).
void rmsnorm_fwd_cuda_launch(const at::Tensor& input2d,
                             const at::Tensor& residual_in2d_or_undef,
                             const at::Tensor& weight_or_undef,
                             at::Tensor& output2d,
                             at::Tensor& residual_out2d_or_undef,
                             at::Tensor& rstd_or_undef,
                             double eps,
                             fused_norm::NormEpilogue epi);

// Launches the fused residual-add + LayerNorm forward (norm_fwd_ln.cu):
//   residual_out = round(input + residual_in); output = layer_norm(residual_out).
// residual_out MAY alias residual_in (in-place). mean/rstd (when defined):
// contiguous (M,) tensors of the accumulation dtype receiving the per-row
// statistics of the ROUNDED sum - what autograd saves. Same 2-D
// preconditions as layernorm_cuda_launch for every tensor.
void fused_add_layernorm_fwd_cuda_launch(const at::Tensor& input2d,
                                         const at::Tensor& residual_in2d,
                                         const at::Tensor& weight_or_undef,
                                         const at::Tensor& bias_or_undef,
                                         at::Tensor& output2d,
                                         at::Tensor& residual_out2d,
                                         at::Tensor& mean_or_undef,
                                         at::Tensor& rstd_or_undef,
                                         double eps);
