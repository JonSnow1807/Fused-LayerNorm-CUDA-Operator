/*
 * PyTorch C++ Extension Bindings for Fused LayerNorm CUDA Operator
 * 
 * This file provides the C++ interface between PyTorch and our CUDA kernels,
 * handling tensor validation, memory management, and autograd integration.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda_runtime.h>

// Forward declarations of CUDA functions
void layernorm_forward_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    at::Tensor mean,
    at::Tensor rstd,
    float epsilon);

// Forward declaration of optimized kernel
void layernorm_forward_cuda_optimized(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    at::Tensor mean,
    at::Tensor rstd,
    float epsilon);

void layernorm_backward_cuda(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor gamma,
    at::Tensor grad_input,
    at::Tensor grad_gamma,
    at::Tensor grad_beta);

// Utility macro for checking CUDA tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward pass implementation
std::vector<at::Tensor> layernorm_forward(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon) {
    
    // Input validation
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, hidden_size)");
    
    if (gamma.defined()) {
        CHECK_INPUT(gamma);
        TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1D");
        TORCH_CHECK(gamma.size(0) == input.size(1), 
                   "Gamma size must match hidden dimension");
    }
    
    if (beta.defined()) {
        CHECK_INPUT(beta);
        TORCH_CHECK(beta.dim() == 1, "Beta must be 1D");
        TORCH_CHECK(beta.size(0) == input.size(1), 
                   "Beta size must match hidden dimension");
    }
    
    // Allocate output tensors
    auto output = at::empty_like(input);
    
    // Use float for accumulation to ensure numerical stability
    auto acc_type = input.scalar_type() == at::ScalarType::Half ? 
                    at::ScalarType::Float : input.scalar_type();
    
    auto mean = at::empty({input.size(0)}, input.options().dtype(acc_type));
    auto rstd = at::empty({input.size(0)}, input.options().dtype(acc_type));
    
    // Get hidden size for deciding which kernel to use
    const int hidden_size = input.size(1);
    
    // Launch CUDA kernel - use optimized version for large hidden sizes
    if (hidden_size >= 4096) {
        layernorm_forward_cuda_optimized(input, gamma, beta, output, mean, rstd, epsilon);
    } else {
        layernorm_forward_cuda(input, gamma, beta, output, mean, rstd, epsilon);
    }
    
    return {output, mean, rstd};
}

// Backward pass implementation
std::vector<at::Tensor> layernorm_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor gamma,
    at::Tensor beta,
    std::array<bool, 3> output_mask) {
    
    // Input validation
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);
    
    TORCH_CHECK(grad_output.sizes() == input.sizes(), 
               "grad_output shape must match input shape");
    
    if (gamma.defined()) {
        CHECK_INPUT(gamma);
    }
    
    // Allocate gradient tensors based on output mask
    at::Tensor grad_input, grad_gamma, grad_beta;
    
    if (output_mask[0]) {
        grad_input = at::empty_like(input);
    }
    
    if (output_mask[1] && gamma.defined()) {
        grad_gamma = at::empty_like(gamma);
    }
    
    if (output_mask[2] && beta.defined()) {
        grad_beta = at::empty_like(beta);
    }
    
    // Launch CUDA kernel
    layernorm_backward_cuda(
        grad_output, input, mean, rstd, gamma,
        grad_input, grad_gamma, grad_beta
    );
    
    return {grad_input, grad_gamma, grad_beta};
}

// Memory usage estimation function
int64_t get_memory_usage(
    int batch_size,
    int hidden_size,
    bool use_mixed_precision) {
    
    // Calculate memory requirements
    int64_t element_size = use_mixed_precision ? 2 : 4;  // FP16 vs FP32
    
    // Forward pass: input + output + mean + rstd
    int64_t forward_memory = 2 * batch_size * hidden_size * element_size +  // input/output
                            2 * batch_size * sizeof(float);  // mean/rstd (always FP32)
    
    // Backward pass: gradients
    int64_t backward_memory = batch_size * hidden_size * element_size +  // grad_input
                             2 * hidden_size * element_size;  // grad_gamma/beta
    
    // Account for temporary workspace
    int64_t workspace = batch_size * 32 * sizeof(float);  // Shared memory per block
    
    return forward_memory + backward_memory + workspace;
}

// Performance hint function
std::string get_performance_hints(
    int batch_size,
    int hidden_size) {
    
    std::stringstream hints;
    
    // Check if dimensions are optimal for GPU
    if (hidden_size % 32 != 0) {
        hints << "Warning: hidden_size is not divisible by 32, may impact performance.\n";
    }
    
    if (hidden_size > 8192) {
        hints << "Info: Large hidden dimension detected. Consider using mixed precision.\n";
    }
    
    if (batch_size < 8) {
        hints << "Info: Small batch size may not fully utilize GPU.\n";
    }
    
    // Estimate occupancy
    int blocks = batch_size;
    int sm_count = 80;  // Assume A100 GPU
    
    if (blocks < sm_count) {
        hints << "Info: Not enough blocks to saturate GPU. Consider batching.\n";
    }
    
    return hints.str();
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused LayerNorm CUDA implementation";
    
    m.def("forward", &layernorm_forward, 
          "LayerNorm forward (CUDA)",
          py::arg("input"),
          py::arg("gamma") = at::Tensor(),
          py::arg("beta") = at::Tensor(),
          py::arg("epsilon") = 1e-5f);
    
    m.def("backward", &layernorm_backward, 
          "LayerNorm backward (CUDA)",
          py::arg("grad_output"),
          py::arg("input"),
          py::arg("mean"),
          py::arg("rstd"),
          py::arg("gamma") = at::Tensor(),
          py::arg("beta") = at::Tensor(),
          py::arg("output_mask") = std::array<bool, 3>{true, true, true});
    
    m.def("get_memory_usage", &get_memory_usage,
          "Estimate memory usage",
          py::arg("batch_size"),
          py::arg("hidden_size"),
          py::arg("use_mixed_precision") = false);
    
    m.def("get_performance_hints", &get_performance_hints,
          "Get performance optimization hints",
          py::arg("batch_size"),
          py::arg("hidden_size"));
    
    // Version information
    m.attr("__version__") = "1.0.0";
    
    // Use CUDART_VERSION instead of CUDA_VERSION
    #ifdef CUDART_VERSION
    m.attr("__cuda_version__") = CUDART_VERSION;
    #else
    m.attr("__cuda_version__") = 0;
    #endif
}