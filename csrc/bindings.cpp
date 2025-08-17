#include <torch/extension.h>

extern "C" {
void layernorm_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    float epsilon);

void layernorm_gelu_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    float epsilon);
}

at::Tensor layernorm(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon) {
    
    auto output = at::empty_like(input);
    layernorm_cuda(input, gamma, beta, output, epsilon);
    return output;
}

at::Tensor layernorm_gelu(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon) {
    
    auto output = at::empty_like(input);
    layernorm_gelu_cuda(input, gamma, beta, output, epsilon);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &layernorm, "LayerNorm forward");
    m.def("layernorm_gelu", &layernorm_gelu, "LayerNorm + GELU");
}
