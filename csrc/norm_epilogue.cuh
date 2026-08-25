// Per-element epilogues applied after normalise+affine, shared by the forward
// kernels. The GELU device functions were extracted verbatim from
// layernorm_cuda_kernel.cu (v0.3.0). The fp8 epilogue functors are added in
// the quant phase of v0.4.0 and documented where they are introduced.
#pragma once

#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FUSED_NORM_HAS_FP8 1
#else
#define FUSED_NORM_HAS_FP8 0
#endif

namespace fused_norm {

// Epilogue functors for the generic forward kernels (norm_fwd_kernels.cuh).
// Each functor defines the element type it stores (out_t) and converts the
// fp32/fp64 normalised value on the way out via store(v, inv_scale)
// (inv_scale is meaningful only for the quantising epilogues; others get
// 1.0f). kNeedsRowMax=true adds a per-row amax pass in the kernel, after
// which finalize_scale() maps the row amax to the stored scale and the
// inv_scale handed to store().
template <typename scalar_t, typename acc_t>
struct EpiNone {
  using out_t = scalar_t;
  static constexpr bool kNeedsRowMax = false;
  __device__ __forceinline__ float load_inv_scale() const { return 1.f; }
  __device__ __forceinline__ out_t store(acc_t v, float /*inv_scale*/) const {
    return static_cast<out_t>(v);
  }
};

#if FUSED_NORM_HAS_FP8

// fp8 E4M3 epilogues (inference-only; fp16/bf16/fp32 inputs). Both round the
// normalised value through scalar_t FIRST and quantise that: the fp8 output
// is then byte-identical to quantising the None-epilogue output (composite
// equivalence - vLLM's fp8 epilogue makes the same deliberate choice), which
// the tests assert. scale is the DEQUANT scale (x ~ fp8.float() * scale).
// E4M3-fn max finite = 448. On sm_80 the conversion intrinsic is emulated
// (no hardware fp8 convert before sm_89) - fine for an epilogue; measured,
// not assumed, in the benchmarks.

__device__ __forceinline__ __nv_fp8_e4m3 to_fp8_e4m3(float v, float inv_scale) {
  // fminf/fmaxf DROP a NaN operand, which would silently quantise NaN inputs
  // to -448; pass NaN through instead (SATFINITE keeps it NaN in e4m3fn),
  // matching torch's .to(float8_e4m3fn) semantics and the composite fallback.
  const float p = v * inv_scale;
  const float q = isnan(p) ? p : fminf(fmaxf(p, -448.f), 448.f);
  // __nv_cvt_float_to_fp8 returns the raw STORAGE byte; assign it to __x
  // directly. The __nv_fp8_e4m3(unsigned char) constructor would instead
  // numerically convert the byte's value - a silent corruption caught by the
  // byte-equality tests.
  __nv_fp8_e4m3 out;
  out.__x = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
  return out;
}

// Static per-tensor scale: a [1] fp32 CUDA tensor dereferenced ON DEVICE (no
// .item(), no host sync - CUDA-graph capturable). Each thread inverts once.
template <typename scalar_t, typename acc_t>
struct EpiFp8Static {
  const float* scale_ptr;  // [1]
  using out_t = __nv_fp8_e4m3;
  static constexpr bool kNeedsRowMax = false;
  __device__ __forceinline__ float load_inv_scale() const { return 1.f / *scale_ptr; }
  __device__ __forceinline__ out_t store(acc_t v, float inv_scale) const {
    const scalar_t rounded = static_cast<scalar_t>(v);
    return to_fp8_e4m3(static_cast<float>(rounded), inv_scale);
  }
};

// Dynamic per-token scale: the kernel's row-max pass feeds the amax of the
// scalar_t-rounded outputs to finalize_scale, which clamps to scale_ub (when
// > 0), guards all-zero rows, stores scale_out[row] and returns 1/scale.
template <typename scalar_t, typename acc_t>
struct EpiFp8Dynamic {
  float* scale_out;  // [M]
  float scale_ub;    // <= 0 => unused
  using out_t = __nv_fp8_e4m3;
  static constexpr bool kNeedsRowMax = true;
  __device__ __forceinline__ float load_inv_scale() const { return 1.f; }  // replaced per row
  __device__ __forceinline__ float finalize_scale(float amax, int64_t row) const {
    // A NaN amax (any NaN in the row) must poison the scale like torch.amax
    // does in the eager composite; fminf/fmaxf would silently drop it.
    if (!isnan(amax)) {
      if (scale_ub > 0.f) amax = fminf(amax, scale_ub);
      amax = fmaxf(amax, 1e-12f);
    }
    const float scale = amax / 448.f;
    scale_out[row] = scale;
    return 1.f / scale;
  }
  __device__ __forceinline__ out_t store(acc_t v, float inv_scale) const {
    const scalar_t rounded = static_cast<scalar_t>(v);
    return to_fp8_e4m3(static_cast<float>(rounded), inv_scale);
  }
};

#endif  // FUSED_NORM_HAS_FP8

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
