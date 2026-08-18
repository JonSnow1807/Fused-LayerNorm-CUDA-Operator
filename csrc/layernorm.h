// Shared declaration between the two translation units of the extension:
//   csrc/bindings.cpp            -- Python bindings, argument validation, reshape to/from 2-D
//   csrc/layernorm_cuda_kernel.cu -- the CUDA kernel and its launcher (defined below)
#pragma once

#include <ATen/ATen.h>

// Launches the fused LayerNorm(+GELU) kernel on the current CUDA stream.
//
// Preconditions (checked by the caller in bindings.cpp, NOT here):
//   * input2d  : contiguous CUDA tensor of shape (M, N), dtype float32/float64/float16/bfloat16
//   * output2d : contiguous CUDA tensor with the same shape/dtype/device as input2d
//   * weight_or_undefined / bias_or_undefined : either an undefined at::Tensor (affine term
//     omitted) or a contiguous 1-D CUDA tensor of length N with input2d's dtype/device.
//     They are independent: weight-only and bias-only are both valid.
//   * eps      : added to the biased variance before rsqrt
//   * gelu     : if true, apply GELU to the normalized value before storing
//   * gelu_tanh: only meaningful when gelu is true; false = erf GELU, true = tanh approximation
//
// M == 0 or N == 0 is a no-op (nothing is launched).
void layernorm_cuda_launch(const at::Tensor& input2d,
                           const at::Tensor& weight_or_undefined,
                           const at::Tensor& bias_or_undefined,
                           at::Tensor& output2d,
                           double eps,
                           bool gelu,
                           bool gelu_tanh);
