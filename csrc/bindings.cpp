// Python bindings for the fused LayerNorm(+GELU) CUDA extension.
//
// This file owns everything that is not the kernel: argument validation, flattening the
// input to a contiguous 2-D (rows, N) view, allocating the output, calling the launcher
// declared in layernorm.h, and reshaping the result back to the input's shape.
//
// Exposed module (name comes from TORCH_EXTENSION_NAME, i.e. "fused_layernorm_cuda"):
//   layernorm(input, weight=None, bias=None, eps=1e-5) -> Tensor
//   layernorm_gelu(input, weight=None, bias=None, eps=1e-5, approximate="none") -> Tensor
//   __version__ == the package version (injected by setup.py at build time)
//
// Both functions are FORWARD ONLY: the returned tensor has no grad_fn.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>  // getCurrentDeviceProperties (backward chunk count)

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Version injected by setup.py as an unquoted token (-DFUSED_LN_VERSION=0.4.0);
// stringified here (pybind11 VERSION_INFO idiom). "unknown" only for builds
// that bypass setup.py.
#ifndef FUSED_LN_VERSION
#define FUSED_LN_VERSION unknown
#endif
#define FLN_STRINGIFY_(x) #x
#define FLN_STRINGIFY(x) FLN_STRINGIFY_(x)
#define FLN_VERSION_STRING FLN_STRINGIFY(FUSED_LN_VERSION)

#include "layernorm.h"

namespace {

// Validates an optional affine parameter against the input. Returns an undefined tensor when
// the parameter was not given, otherwise a contiguous tensor. `name` is "weight" or "bias" and
// is used in every error message so the caller can tell which argument is wrong.
at::Tensor check_affine(const std::optional<at::Tensor>& opt,
                        const char* name,
                        const at::Tensor& input,
                        int64_t N) {
  if (!opt.has_value()) {
    return at::Tensor();  // undefined => "not provided"
  }
  const at::Tensor& t = *opt;
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor, but it is on ", t.device());
  TORCH_CHECK(t.dim() == 1, name, " must be 1-D, got a ", t.dim(), "-D tensor");
  TORCH_CHECK(t.size(0) == N, name, " must have length input.shape[-1] = ", N, ", got ",
              t.size(0));
  TORCH_CHECK(t.scalar_type() == input.scalar_type(), name, " dtype (", t.scalar_type(),
              ") must match input dtype (", input.scalar_type(), ")");
  TORCH_CHECK(t.device() == input.device(), name, " must be on the same device as input (",
              name, " is on ", t.device(), ", input is on ", input.device(), ")");
  return t.contiguous();
}

// Shared implementation of both entry points.
at::Tensor layernorm_impl(const at::Tensor& input,
                          const std::optional<at::Tensor>& weight,
                          const std::optional<at::Tensor>& bias,
                          double eps,
                          bool gelu,
                          bool gelu_tanh) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1,
              "input must have rank >= 1 (normalization is over the last dimension), "
              "got a 0-d tensor");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
              "input dtype must be float32, float64, float16 or bfloat16, got ", st);
  const int64_t N = input.size(-1);

  const at::Tensor w = check_affine(weight, "weight", input, N);
  const at::Tensor b = check_affine(bias, "bias", input, N);

  // Nothing to normalize (M == 0 rows, or N == 0 columns): return an empty result without
  // launching a kernel (a zero-sized grid would be a launch error).
  if (input.numel() == 0) {
    return at::empty_like(input);
  }
  const int64_t M = input.numel() / N;  // N > 0 here because numel > 0

  // Flatten leading dims into rows. contiguous() is a no-op for already-contiguous input.
  const at::Tensor x2d = input.contiguous().view({M, N});
  at::Tensor y2d = at::empty_like(x2d);  // same dtype/device, contiguous (M, N)

  using fused_norm::NormEpilogue;
  const NormEpilogue epi = !gelu ? NormEpilogue::kNone
                                 : (gelu_tanh ? NormEpilogue::kGeluTanh : NormEpilogue::kGeluErf);
  layernorm_cuda_launch(x2d, w, b, y2d, eps, epi);

  return y2d.view(input.sizes());  // back to the caller's shape
}

at::Tensor layernorm(const at::Tensor& input,
                     const std::optional<at::Tensor>& weight,
                     const std::optional<at::Tensor>& bias,
                     double eps) {
  return layernorm_impl(input, weight, bias, eps, /*gelu=*/false, /*gelu_tanh=*/false);
}

// Shared implementation of the fused residual-add + norm entry points.
//   residual_out = round(input + residual); out = norm(residual_out)
// Returns (out, residual_out). inplace=true writes the sum into `residual`'s
// storage (which must be contiguous - a silent .contiguous() copy would make
// the mutation invisible to the caller) and is inference-only.
std::tuple<at::Tensor, at::Tensor> fused_add_impl(const at::Tensor& input,
                                                  at::Tensor& residual,
                                                  const std::optional<at::Tensor>& weight,
                                                  const std::optional<at::Tensor>& bias,
                                                  double eps,
                                                  bool inplace,
                                                  bool rms,
                                                  at::Tensor* mean_out = nullptr,
                                                  at::Tensor* rstd_out = nullptr) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1,
              "input must have rank >= 1 (normalization is over the last dimension), "
              "got a 0-d tensor");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
              "input dtype must be float32, float64, float16 or bfloat16, got ", st);
  TORCH_CHECK(residual.sizes() == input.sizes(), "residual must have input's shape (",
              input.sizes(), "), got ", residual.sizes());
  TORCH_CHECK(residual.scalar_type() == st, "residual dtype (", residual.scalar_type(),
              ") must match input dtype (", st, ")");
  TORCH_CHECK(residual.device() == input.device(),
              "residual must be on the same device as input");
  const int64_t N = input.size(-1);

  const at::Tensor w = check_affine(weight, "weight", input, N);
  const at::Tensor b = check_affine(bias, "bias", input, N);

  if (inplace) {
    TORCH_CHECK(residual.is_contiguous(),
                "inplace fused_add requires a contiguous residual (a hidden copy would "
                "make the in-place update invisible); pass inplace=False or call "
                ".contiguous() yourself");
    TORCH_CHECK(!(at::GradMode::is_enabled() &&
                  (input.requires_grad() || residual.requires_grad() ||
                   (w.defined() && w.requires_grad()) || (b.defined() && b.requires_grad()))),
                "inplace fused_add is inference-only; use the out-of-place op under autograd");
  }

  const auto acc = input.options().dtype(input.scalar_type() == at::kDouble ? at::kDouble
                                                                            : at::kFloat);
  auto leading = input.sizes().vec();
  leading.pop_back();

  if (input.numel() == 0) {
    at::Tensor z = inplace ? residual : at::empty_like(input);
    if (mean_out != nullptr) *mean_out = at::empty(leading, acc);
    if (rstd_out != nullptr) *rstd_out = at::empty(leading, acc);
    return {at::empty_like(input), z};
  }
  const int64_t M = input.numel() / N;

  const at::Tensor x2d = input.contiguous().view({M, N});
  const at::Tensor r2d = residual.contiguous().view({M, N});  // no-op copy when inplace (checked)
  at::Tensor y2d = at::empty_like(x2d);
  at::Tensor z2d = inplace ? r2d : at::empty_like(x2d);
  at::Tensor mean = mean_out != nullptr ? at::empty({M}, acc) : at::Tensor();
  at::Tensor rstd = rstd_out != nullptr ? at::empty({M}, acc) : at::Tensor();

  if (rms) {
    rmsnorm_fwd_cuda_launch(x2d, r2d, w, y2d, z2d, rstd, eps,
                            fused_norm::NormEpilogue::kNone);
  } else {
    fused_add_layernorm_fwd_cuda_launch(x2d, r2d, w, b, y2d, z2d, mean, rstd, eps);
  }

  if (mean_out != nullptr) *mean_out = mean.view(leading);
  if (rstd_out != nullptr) *rstd_out = rstd.view(leading);
  at::Tensor z_out = inplace ? residual : z2d.view(input.sizes());
  return {y2d.view(input.sizes()), z_out};
}

std::tuple<at::Tensor, at::Tensor> fused_add_layernorm(const at::Tensor& input,
                                                       at::Tensor& residual,
                                                       const std::optional<at::Tensor>& weight,
                                                       const std::optional<at::Tensor>& bias,
                                                       double eps,
                                                       bool inplace) {
  return fused_add_impl(input, residual, weight, bias, eps, inplace, /*rms=*/false);
}

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm(const at::Tensor& input,
                                                     at::Tensor& residual,
                                                     const std::optional<at::Tensor>& weight,
                                                     double eps,
                                                     bool inplace) {
  return fused_add_impl(input, residual, weight, std::nullopt, eps, inplace, /*rms=*/true);
}

at::ScalarType acc_dtype_for(at::ScalarType st) {
  return st == at::kDouble ? at::kDouble : at::kFloat;
}

// Leading dims of `input` (its shape minus the last axis) - the shape the
// per-row statistics are returned in.
std::vector<int64_t> leading_sizes(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  sizes.pop_back();
  return sizes;
}

// Training forwards: like the inference entry points but also return the
// per-row statistics autograd saves (acc dtype: fp32, or fp64 for double).
std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_fwd_train(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1, "input must have rank >= 1");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
              "input dtype must be float32, float64, float16 or bfloat16, got ", st);
  const int64_t N = input.size(-1);
  const at::Tensor w = check_affine(weight, "weight", input, N);
  const at::Tensor b = check_affine(bias, "bias", input, N);

  const auto acc = input.options().dtype(acc_dtype_for(st));
  if (input.numel() == 0) {
    return {at::empty_like(input), at::empty(leading_sizes(input), acc),
            at::empty(leading_sizes(input), acc)};
  }
  const int64_t M = input.numel() / N;
  const at::Tensor x2d = input.contiguous().view({M, N});
  at::Tensor y2d = at::empty_like(x2d);
  at::Tensor mean = at::empty({M}, acc);
  at::Tensor rstd = at::empty({M}, acc);
  layernorm_fwd_train_cuda_launch(x2d, w, b, y2d, mean, rstd, eps);
  return {y2d.view(input.sizes()), mean.view(leading_sizes(input)),
          rstd.view(leading_sizes(input))};
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_fwd_train(const at::Tensor& input,
                                                     const std::optional<at::Tensor>& weight,
                                                     double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1, "input must have rank >= 1");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
              "input dtype must be float32, float64, float16 or bfloat16, got ", st);
  const int64_t N = input.size(-1);
  const at::Tensor w = check_affine(weight, "weight", input, N);

  const auto acc = input.options().dtype(acc_dtype_for(st));
  if (input.numel() == 0) {
    return {at::empty_like(input), at::empty(leading_sizes(input), acc)};
  }
  const int64_t M = input.numel() / N;
  const at::Tensor x2d = input.contiguous().view({M, N});
  at::Tensor y2d = at::empty_like(x2d);
  at::Tensor rstd = at::empty({M}, acc);
  at::Tensor undef;
  rmsnorm_fwd_cuda_launch(x2d, /*residual_in=*/undef, w, y2d, /*residual_out=*/undef, rstd,
                          eps, fused_norm::NormEpilogue::kNone);
  return {y2d.view(input.sizes()), rstd.view(leading_sizes(input))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_add_layernorm_fwd_train(
    const at::Tensor& input,
    const at::Tensor& residual,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  at::Tensor res = residual;  // out-of-place only: residual is never mutated here
  at::Tensor mean, rstd;
  auto [y, z] = fused_add_impl(input, res, weight, bias, eps, /*inplace=*/false,
                               /*rms=*/false, &mean, &rstd);
  return {y, z, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_add_rmsnorm_fwd_train(
    const at::Tensor& input,
    const at::Tensor& residual,
    const std::optional<at::Tensor>& weight,
    double eps) {
  at::Tensor res = residual;
  at::Tensor rstd;
  auto [y, z] = fused_add_impl(input, res, weight, std::nullopt, eps, /*inplace=*/false,
                               /*rms=*/true, /*mean_out=*/nullptr, &rstd);
  return {y, z, rstd};
}

// Backward entry points. xz is what the forward normalised; dz_extra (when
// given) is the downstream cotangent of the fused-add new_residual output and
// is folded into dx (dx == dresidual for fused-add). Unrequested grads come
// back as empty 0-element tensors (custom-op schemas need fixed arity);
// Python maps them to None.
std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_bwd(
    const at::Tensor& dy,
    const at::Tensor& xz,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& dz_extra,
    bool need_dx,
    bool need_dgamma,
    bool need_dbeta) {
  const int64_t N = xz.size(-1);
  const int64_t M = xz.numel() == 0 ? 0 : xz.numel() / N;
  const at::Tensor w = check_affine(weight, "weight", xz, N);
  const at::Tensor xz2d = xz.contiguous().view({M, N});
  const at::Tensor dy2d = dy.contiguous().view({M, N});
  const at::Tensor mean1d = mean.contiguous().view({M});
  const at::Tensor rstd1d = rstd.contiguous().view({M});
  at::Tensor dz2d;
  if (dz_extra.has_value()) dz2d = dz_extra->contiguous().view({M, N});

  const auto acc = xz.options().dtype(acc_dtype_for(xz.scalar_type()));
  // Each unrequested grad gets its OWN empty tensor: custom-op outputs may
  // not alias each other.
  at::Tensor dx = at::empty({0}, xz.options());
  if (need_dx) {
    at::Tensor dx2d = at::empty_like(xz2d);
    norm_bwd_dx_cuda_launch(/*rms=*/false, dy2d, dz2d, xz2d, mean1d, rstd1d, w, dx2d);
    dx = dx2d.view(xz.sizes());
  }

  at::Tensor dgamma = at::empty({0}, xz.options());
  at::Tensor dbeta = at::empty({0}, xz.options());
  if ((need_dgamma || need_dbeta) && M > 0) {
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t chunks = std::clamp<int64_t>((M + 255) / 256, 1, 4 * sms);
    at::Tensor dg_part = at::empty({chunks, N}, acc);
    at::Tensor db_part = need_dbeta ? at::empty({chunks, N}, acc) : at::Tensor();
    norm_bwd_param_partials_cuda_launch(/*rms=*/false, dy2d, xz2d, mean1d, rstd1d, dg_part,
                                        db_part);
    if (need_dgamma) dgamma = dg_part.sum(0).to(xz.scalar_type());
    if (need_dbeta) dbeta = db_part.sum(0).to(xz.scalar_type());
  }
  return {dx, dgamma, dbeta};
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_bwd(const at::Tensor& dy,
                                               const at::Tensor& xz,
                                               const at::Tensor& rstd,
                                               const std::optional<at::Tensor>& weight,
                                               const std::optional<at::Tensor>& dz_extra,
                                               bool need_dx,
                                               bool need_dgamma) {
  const int64_t N = xz.size(-1);
  const int64_t M = xz.numel() == 0 ? 0 : xz.numel() / N;
  const at::Tensor w = check_affine(weight, "weight", xz, N);
  const at::Tensor xz2d = xz.contiguous().view({M, N});
  const at::Tensor dy2d = dy.contiguous().view({M, N});
  const at::Tensor rstd1d = rstd.contiguous().view({M});
  at::Tensor dz2d;
  if (dz_extra.has_value()) dz2d = dz_extra->contiguous().view({M, N});
  at::Tensor undef;

  const auto acc = xz.options().dtype(acc_dtype_for(xz.scalar_type()));
  at::Tensor dx = at::empty({0}, xz.options());
  if (need_dx) {
    at::Tensor dx2d = at::empty_like(xz2d);
    norm_bwd_dx_cuda_launch(/*rms=*/true, dy2d, dz2d, xz2d, /*mean=*/undef, rstd1d, w, dx2d);
    dx = dx2d.view(xz.sizes());
  }

  at::Tensor dgamma = at::empty({0}, xz.options());
  if (need_dgamma && M > 0) {
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t chunks = std::clamp<int64_t>((M + 255) / 256, 1, 4 * sms);
    at::Tensor dg_part = at::empty({chunks, N}, acc);
    at::Tensor db_undef;
    norm_bwd_param_partials_cuda_launch(/*rms=*/true, dy2d, xz2d, /*mean=*/undef, rstd1d,
                                        dg_part, db_undef);
    dgamma = dg_part.sum(0).to(xz.scalar_type());
  }
  return {dx, dgamma};
}

// Shared implementation of the fp8-output RMSNorm entry points (plain and
// fused-add, static and dynamic scale). Inference-only. Returns
// (out_fp8, residual_out_or_undef, scale_out_or_undef).
std::tuple<at::Tensor, at::Tensor, at::Tensor> rmsnorm_fp8_impl(
    const at::Tensor& input,
    const std::optional<at::Tensor>& residual,  // fused-add iff given
    const std::optional<at::Tensor>& weight,
    double eps,
    const std::optional<at::Tensor>& scale,     // static iff given, else dynamic
    double scale_ub,                             // dynamic only; <= 0 => unused
    bool inplace) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1, "input must have rank >= 1");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kHalf || st == at::kBFloat16,
              "fp8 ops support float32, float16 or bfloat16 inputs, got ", st);
  const int64_t N = input.size(-1);
  const at::Tensor w = check_affine(weight, "weight", input, N);

  const bool fused_add = residual.has_value();
  at::Tensor res;
  if (fused_add) {
    res = *residual;
    TORCH_CHECK(res.sizes() == input.sizes() && res.scalar_type() == st &&
                    res.device() == input.device(),
                "residual must match input's shape/dtype/device");
    if (inplace) {
      TORCH_CHECK(res.is_contiguous(),
                  "inplace fused_add requires a contiguous residual");
    }
  } else {
    TORCH_CHECK(!inplace, "inplace requires a residual");
  }
  TORCH_CHECK(!(at::GradMode::is_enabled() &&
                (input.requires_grad() || (fused_add && res.requires_grad()) ||
                 (w.defined() && w.requires_grad()))),
              "fp8 norm ops are inference-only");

  auto leading = input.sizes().vec();
  leading.pop_back();
  const auto out_opts = input.options().dtype(at::kFloat8_e4m3fn);
  const auto f32_opts = input.options().dtype(at::kFloat);
  const bool dynamic = !scale.has_value();

  auto scale_shape = leading;
  scale_shape.push_back(1);  // trailing broadcast dim: out.float() * scale dequantises

  if (input.numel() == 0) {
    at::Tensor z = fused_add ? (inplace ? res : at::empty_like(input)) : at::Tensor();
    at::Tensor s = dynamic ? at::empty(scale_shape, f32_opts) : at::Tensor();
    return {at::empty(input.sizes(), out_opts), z, s};
  }
  const int64_t M = input.numel() / N;

  const at::Tensor x2d = input.contiguous().view({M, N});
  at::Tensor r2d, z2d;
  if (fused_add) {
    r2d = res.contiguous().view({M, N});
    z2d = inplace ? r2d : at::empty_like(x2d);
  }
  at::Tensor y2d = at::empty({M, N}, out_opts);
  at::Tensor undef;

  fused_norm::EpilogueParams params;
  fused_norm::NormEpilogue epi;
  at::Tensor scale_out;
  if (dynamic) {
    epi = fused_norm::NormEpilogue::kFp8Dynamic;
    scale_out = at::empty({M}, f32_opts);
    params.scale_out = scale_out;
    params.scale_ub = scale_ub;
  } else {
    epi = fused_norm::NormEpilogue::kFp8Static;
    TORCH_CHECK(scale->is_cuda() && scale->scalar_type() == at::kFloat && scale->numel() == 1,
                "static fp8 scale must be a 1-element fp32 CUDA tensor");
    params.scale_in = scale->contiguous();
  }

  rmsnorm_fwd_cuda_launch(x2d, r2d, w, y2d, z2d, /*rstd=*/undef, eps, epi, params);

  at::Tensor z_out;
  if (fused_add) z_out = inplace ? res : z2d.view(input.sizes());
  at::Tensor s_out;
  if (dynamic) s_out = scale_out.view(scale_shape);
  return {y2d.view(input.sizes()), z_out, s_out};
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_fp8_static(const at::Tensor& input,
                                                      const at::Tensor& scale,
                                                      const std::optional<at::Tensor>& weight,
                                                      double eps) {
  auto [y, z, s] = rmsnorm_fp8_impl(input, std::nullopt, weight, eps, scale, 0.0, false);
  return {y, scale};
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_fp8_dynamic(const at::Tensor& input,
                                                       const std::optional<at::Tensor>& weight,
                                                       double eps,
                                                       const std::optional<double>& scale_ub) {
  auto [y, z, s] =
      rmsnorm_fp8_impl(input, std::nullopt, weight, eps, std::nullopt,
                       scale_ub.value_or(0.0), false);
  return {y, s};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_add_rmsnorm_fp8_static(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& weight,
    double eps,
    bool inplace) {
  auto [y, z, s] = rmsnorm_fp8_impl(input, residual, weight, eps, scale, 0.0, inplace);
  return {y, z, scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_add_rmsnorm_fp8_dynamic(
    const at::Tensor& input,
    const at::Tensor& residual,
    const std::optional<at::Tensor>& weight,
    double eps,
    const std::optional<double>& scale_ub,
    bool inplace) {
  return rmsnorm_fp8_impl(input, residual, weight, eps, std::nullopt,
                          scale_ub.value_or(0.0), inplace);
}

// RMSNorm over the last dimension: y = x * rsqrt(mean(x^2) + eps) [* weight].
// No bias (RMSNorm has none); eps must already be resolved by the caller
// (the F.rms_norm eps=None machine-epsilon convention lives in Python).
at::Tensor rmsnorm(const at::Tensor& input,
                   const std::optional<at::Tensor>& weight,
                   double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor, but it is on ", input.device());
  TORCH_CHECK(input.dim() >= 1,
              "input must have rank >= 1 (normalization is over the last dimension), "
              "got a 0-d tensor");
  const auto st = input.scalar_type();
  TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
              "input dtype must be float32, float64, float16 or bfloat16, got ", st);
  const int64_t N = input.size(-1);

  const at::Tensor w = check_affine(weight, "weight", input, N);

  if (input.numel() == 0) {
    return at::empty_like(input);
  }
  const int64_t M = input.numel() / N;

  const at::Tensor x2d = input.contiguous().view({M, N});
  at::Tensor y2d = at::empty_like(x2d);

  at::Tensor undef;
  rmsnorm_fwd_cuda_launch(x2d, /*residual_in=*/undef, w, y2d, /*residual_out=*/undef,
                          /*rstd=*/undef, eps, fused_norm::NormEpilogue::kNone);

  return y2d.view(input.sizes());
}

at::Tensor layernorm_gelu(const at::Tensor& input,
                          const std::optional<at::Tensor>& weight,
                          const std::optional<at::Tensor>& bias,
                          double eps,
                          const std::string& approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
              "approximate must be 'none' or 'tanh', got '", approximate, "'");
  return layernorm_impl(input, weight, bias, eps, /*gelu=*/true,
                        /*gelu_tanh=*/approximate == "tanh");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Fused LayerNorm (+GELU) forward CUDA kernel. Forward only: no autograd.";

  m.def("layernorm", &layernorm,
        "LayerNorm over the last dimension of a CUDA tensor.\n\n"
        "Equivalent to torch.nn.functional.layer_norm(input, (input.shape[-1],), weight, bias, eps).\n"
        "input: CUDA tensor, rank >= 1, float32/float64/float16/bfloat16 (made contiguous internally).\n"
        "weight, bias: optional 1-D tensors of length input.shape[-1] with input's dtype/device;\n"
        "each may be None independently.\n"
        "Returns a new tensor with input's shape/dtype/device.\n"
        "Forward only: the result has no grad_fn.",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5);

  m.def("layernorm_gelu", &layernorm_gelu,
        "LayerNorm over the last dimension followed by GELU, in one kernel.\n\n"
        "Equivalent to F.gelu(F.layer_norm(input, (input.shape[-1],), weight, bias, eps),\n"
        "approximate=approximate) with approximate in {'none' (erf), 'tanh'}.\n"
        "Same argument rules as layernorm(). Forward only: the result has no grad_fn.",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5, py::arg("approximate") = "none");

  m.def("rmsnorm", &rmsnorm,
        "RMSNorm over the last dimension of a CUDA tensor.\n\n"
        "Equivalent to torch.nn.functional.rms_norm(input, (input.shape[-1],), weight, eps)\n"
        "with a concrete eps (resolve eps=None on the Python side).\n"
        "input: CUDA tensor, rank >= 1, float32/float64/float16/bfloat16.\n"
        "weight: optional 1-D tensor of length input.shape[-1] with input's dtype/device.\n"
        "Forward only from this entry point: the result has no grad_fn.",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("eps") = 1e-6);

  m.def("fused_add_layernorm", &fused_add_layernorm,
        "Fused residual-add + LayerNorm: residual_out = input + residual (rounded once);\n"
        "out = layer_norm(residual_out). Returns (out, residual_out). With inplace=True the\n"
        "sum is written into `residual`'s storage (contiguous required; inference-only).\n"
        "Statistics are computed over the rounded sum, so `out` equals a plain layernorm of\n"
        "residual_out bitwise. Forward only from this entry point.",
        py::arg("input"), py::arg("residual"), py::arg("weight") = py::none(),
        py::arg("bias") = py::none(), py::arg("eps") = 1e-5, py::arg("inplace") = false);

  m.def("fused_add_rmsnorm", &fused_add_rmsnorm,
        "Fused residual-add + RMSNorm: residual_out = input + residual (rounded once);\n"
        "out = rms_norm(residual_out). Returns (out, residual_out). Same inplace/aliasing\n"
        "rules as fused_add_layernorm; eps must be concrete (resolve eps=None in Python).",
        py::arg("input"), py::arg("residual"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6, py::arg("inplace") = false);

  m.def("layernorm_fwd_train", &layernorm_fwd_train,
        "LayerNorm forward that also returns (mean, rstd) per row (acc dtype) for autograd.\n"
        "Output is bitwise identical to layernorm().",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5);

  m.def("rmsnorm_fwd_train", &rmsnorm_fwd_train,
        "RMSNorm forward that also returns rstd per row (acc dtype) for autograd.",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("eps") = 1e-6);

  m.def("fused_add_layernorm_fwd_train", &fused_add_layernorm_fwd_train,
        "Out-of-place fused_add_layernorm returning (out, residual_out, mean, rstd).",
        py::arg("input"), py::arg("residual"), py::arg("weight") = py::none(),
        py::arg("bias") = py::none(), py::arg("eps") = 1e-5);

  m.def("fused_add_rmsnorm_fwd_train", &fused_add_rmsnorm_fwd_train,
        "Out-of-place fused_add_rmsnorm returning (out, residual_out, rstd).",
        py::arg("input"), py::arg("residual"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6);

  m.def("layernorm_bwd", &layernorm_bwd,
        "LayerNorm backward: returns (dx, dgamma, dbeta); unrequested grads are empty\n"
        "0-element tensors. dz_extra (fused-add) is folded into dx (= dresidual).\n"
        "Parameter grads use a deterministic two-stage reduction (bitwise reproducible).",
        py::arg("dy"), py::arg("xz"), py::arg("mean"), py::arg("rstd"),
        py::arg("weight") = py::none(), py::arg("dz_extra") = py::none(),
        py::arg("need_dx") = true, py::arg("need_dgamma") = true, py::arg("need_dbeta") = true);

  m.def("rmsnorm_bwd", &rmsnorm_bwd,
        "RMSNorm backward: returns (dx, dgamma); same conventions as layernorm_bwd.",
        py::arg("dy"), py::arg("xz"), py::arg("rstd"), py::arg("weight") = py::none(),
        py::arg("dz_extra") = py::none(), py::arg("need_dx") = true,
        py::arg("need_dgamma") = true);

  m.def("rmsnorm_fp8_static", &rmsnorm_fp8_static,
        "RMSNorm with fused fp8-E4M3 output, per-tensor dequant scale ([1] fp32 CUDA\n"
        "tensor, read on device - graph-capturable). Returns (out_fp8, scale).\n"
        "The fp8 bytes equal quantising rmsnorm()'s own output (composite equivalence).\n"
        "Inference-only.",
        py::arg("input"), py::arg("scale"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6);

  m.def("rmsnorm_fp8_dynamic", &rmsnorm_fp8_dynamic,
        "RMSNorm with fused fp8-E4M3 output and per-row dynamic scale (amax/448,\n"
        "optionally clamped to scale_ub). Returns (out_fp8, scale[leading dims]).\n"
        "Inference-only.",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("eps") = 1e-6,
        py::arg("scale_ub") = py::none());

  m.def("fused_add_rmsnorm_fp8_static", &fused_add_rmsnorm_fp8_static,
        "fused_add_rmsnorm with fp8-E4M3 output (static scale). Returns\n"
        "(out_fp8, residual_out, scale). Same inplace rules as fused_add_rmsnorm.",
        py::arg("input"), py::arg("residual"), py::arg("scale"),
        py::arg("weight") = py::none(), py::arg("eps") = 1e-6, py::arg("inplace") = false);

  m.def("fused_add_rmsnorm_fp8_dynamic", &fused_add_rmsnorm_fp8_dynamic,
        "fused_add_rmsnorm with fp8-E4M3 output (dynamic per-row scale). Returns\n"
        "(out_fp8, residual_out, scale[leading dims]).",
        py::arg("input"), py::arg("residual"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6, py::arg("scale_ub") = py::none(), py::arg("inplace") = false);

  m.attr("__version__") = FLN_VERSION_STRING;
}
