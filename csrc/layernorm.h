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
