// Per-element epilogues applied after normalise+affine, shared by the forward
// kernels. The GELU device functions were extracted verbatim from
// layernorm_cuda_kernel.cu (v0.3.0). The fp8 epilogue functors are added in
// the quant phase of v0.4.0 and documented where they are introduced.
#pragma once

#include <cuda_runtime.h>

namespace fused_norm {

// Epilogue functors for the generic forward kernels (norm_fwd_kernels.cuh).
// Each functor defines the element type it stores (out_t) and converts the
// fp32/fp64 normalised value on the way out. The fp8 functors (which also set
// kNeedsRowMax and quantise) are added with the quant kernels.
template <typename scalar_t, typename acc_t>
struct EpiNone {
  using out_t = scalar_t;
  static constexpr bool kNeedsRowMax = false;
  __device__ __forceinline__ out_t store(acc_t v) const { return static_cast<out_t>(v); }
};

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

}  // namespace fused_norm
