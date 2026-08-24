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

#include <optional>
#include <string>

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

  m.attr("__version__") = FLN_VERSION_STRING;
}
