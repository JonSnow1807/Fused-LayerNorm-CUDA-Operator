/*
 * Header file for Fused LayerNorm CUDA operators
 * 
 * Provides function declarations and common definitions used across
 * the CUDA implementation and C++ bindings.
 */

#ifndef FUSED_LAYERNORM_H
#define FUSED_LAYERNORM_H

#include <torch/extension.h>
#include <vector>

// CUDA kernel launch functions
void layernorm_forward_cuda(
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

// C++ interface functions
std::vector<at::Tensor> layernorm_forward(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon);

std::vector<at::Tensor> layernorm_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor gamma,
    at::Tensor beta,
    std::array<bool, 3> output_mask);

// Utility functions
int64_t get_memory_usage(
    int batch_size,
    int hidden_size,
    bool use_mixed_precision);

std::string get_performance_hints(
    int batch_size,
    int hidden_size);

// Configuration constants
constexpr float DEFAULT_EPSILON = 1e-5f;
constexpr int MAX_GRID_SIZE = 65535;
constexpr int PREFERRED_BLOCK_SIZE = 256;

// Performance profiling utilities
struct KernelProfile {
    float forward_time_ms;
    float backward_time_ms;
    int64_t memory_allocated;
    int64_t memory_reserved;
};

#endif // FUSED_LAYERNORM_H