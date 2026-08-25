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
  kFp8Static = 3,   // out dtype float8_e4m3fn, per-tensor scale read on device
  kFp8Dynamic = 4,  // out dtype float8_e4m3fn, per-row scale written by the kernel
};

// Runtime state for the quantising epilogues. Only the fields of the selected
// epilogue are read.
struct EpilogueParams {
  at::Tensor scale_in;   // kFp8Static: [1] fp32 CUDA tensor (dequant scale)
  at::Tensor scale_out;  // kFp8Dynamic: [M] fp32, written per row
  double scale_ub = 0.0;  // kFp8Dynamic: clamp for the row amax; <= 0 => unused
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
// epi: kNone, or kFp8Static/kFp8Dynamic (then output2d has dtype
// float8_e4m3fn, scalar dtype comes from input2d, and epi_params carries the
// scale tensor(s); fp64 inputs are rejected for the fp8 epilogues).
void rmsnorm_fwd_cuda_launch(const at::Tensor& input2d,
                             const at::Tensor& residual_in2d_or_undef,
                             const at::Tensor& weight_or_undef,
                             at::Tensor& output2d,
                             at::Tensor& residual_out2d_or_undef,
                             at::Tensor& rstd_or_undef,
                             double eps,
                             fused_norm::NormEpilogue epi,
                             const fused_norm::EpilogueParams& epi_params = {});

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

// Plain LayerNorm forward that also writes per-row mean/rstd (the training
// forward). Runs the generic template in norm_fwd_ln.cu; its normalise
// arithmetic and reduction order match layernorm_cuda_launch's kernels, so
// the output is bitwise identical to the inference path.
void layernorm_fwd_train_cuda_launch(const at::Tensor& input2d,
                                     const at::Tensor& weight_or_undef,
                                     const at::Tensor& bias_or_undef,
                                     at::Tensor& output2d,
                                     at::Tensor& mean2d,
                                     at::Tensor& rstd2d,
                                     double eps);

// Backward (norm_bwd.cu). xz2d is what the forward normalised (input, or the
// rounded sum for fused-add); mean is defined iff LayerNorm; dz_extra (when
// defined) is the downstream cotangent of the fused-add op's new_residual
// output, added elementwise into dx (dx = dresidual for fused-add).
void norm_bwd_dx_cuda_launch(bool rms,
                             const at::Tensor& dy2d,
                             const at::Tensor& dz_extra2d_or_undef,
                             const at::Tensor& xz2d,
                             const at::Tensor& mean_or_undef,
                             const at::Tensor& rstd,
                             const at::Tensor& weight_or_undef,
                             at::Tensor& dx2d);

// Stage 1 of the deterministic two-stage parameter gradients: fixed-chunk
// fp32 partials of shape [num_chunks, N] (dgamma_partials.size(0) chooses the
// chunk count); stage 2 is norm_bwd_param_finalize_cuda_launch. No atomics:
// grads are bitwise run-to-run reproducible.
void norm_bwd_param_partials_cuda_launch(bool rms,
                                         const at::Tensor& dy2d,
                                         const at::Tensor& xz2d,
                                         const at::Tensor& mean_or_undef,
                                         const at::Tensor& rstd,
                                         at::Tensor& dgamma_partials,
                                         at::Tensor& dbeta_partials_or_undef);

// Stage 2: sums each requested [chunks, N] partials tensor over chunks in
// fixed ascending order and casts into the (pre-allocated, param-dtype)
// outputs — both parameters in a single launch, bitwise deterministic. At
// least one of dgamma/dbeta must be defined; an undefined output (or an
// undefined dbeta_partials) is skipped.
void norm_bwd_param_finalize_cuda_launch(const at::Tensor& dgamma_partials,
                                         const at::Tensor& dbeta_partials_or_undef,
                                         at::Tensor& dgamma_or_undef,
                                         at::Tensor& dbeta_or_undef);
