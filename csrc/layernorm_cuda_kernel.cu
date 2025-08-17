#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Simple GELU activation
__device__ __forceinline__ float gelu(float x) {
    const float s = sqrtf(2.0f / M_PI);
    const float a = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(s * (x + a * x3)));
}

// Simple, fast LayerNorm - works on ANY dimension
template<typename scalar_t>
__global__ void layernorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int N,
    float epsilon) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const scalar_t* X = input + row * N;
    scalar_t* Y = output + row * N;
    
    __shared__ float s_mean;
    __shared__ float s_rstd;
    
    // Compute mean - handles any N
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += X[i];
    }
    sum = blockReduceSum(sum);
    
    if (tid == 0) {
        s_mean = sum / N;
    }
    __syncthreads();
    
    // Compute variance - handles any N
    float mean = s_mean;
    sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = X[i] - mean;
        sum += diff * diff;
    }
    sum = blockReduceSum(sum);
    
    if (tid == 0) {
        s_rstd = rsqrtf(sum / N + epsilon);
    }
    __syncthreads();
    
    // Apply normalization - handles any N
    float rstd = s_rstd;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = (X[i] - mean) * rstd;
        if (gamma && beta) {
            val = val * gamma[i] + beta[i];
        }
        Y[i] = val;
    }
}

// Fused LayerNorm + GELU
template<typename scalar_t>
__global__ void layernorm_gelu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int N,
    float epsilon) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const scalar_t* X = input + row * N;
    scalar_t* Y = output + row * N;
    
    __shared__ float s_mean;
    __shared__ float s_rstd;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += X[i];
    }
    sum = blockReduceSum(sum);
    
    if (tid == 0) {
        s_mean = sum / N;
    }
    __syncthreads();
    
    // Compute variance
    float mean = s_mean;
    sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = X[i] - mean;
        sum += diff * diff;
    }
    sum = blockReduceSum(sum);
    
    if (tid == 0) {
        s_rstd = rsqrtf(sum / N + epsilon);
    }
    __syncthreads();
    
    // Apply normalization + GELU
    float rstd = s_rstd;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = (X[i] - mean) * rstd;
        if (gamma && beta) {
            val = val * gamma[i] + beta[i];
        }
        Y[i] = gelu(val);  // Apply GELU
    }
}

extern "C" {

void layernorm_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    float epsilon) {
    
    const int batch = input.size(0);
    const int N = input.size(1);
    
    // Adaptive block size based on hidden dimension
    int threads = 256;
    if (N >= 1024) threads = 512;
    if (N >= 4096) threads = 1024;
    
    dim3 grid(batch);
    dim3 block(threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layernorm", [&] {
        layernorm_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            beta.defined() ? beta.data_ptr<scalar_t>() : nullptr,
            N, epsilon);
    });
}

void layernorm_gelu_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    float epsilon) {
    
    const int batch = input.size(0);
    const int N = input.size(1);
    
    // Adaptive block size
    int threads = 256;
    if (N >= 1024) threads = 512;
    if (N >= 4096) threads = 1024;
    
    dim3 grid(batch);
    dim3 block(threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layernorm_gelu", [&] {
        layernorm_gelu_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            beta.defined() ? beta.data_ptr<scalar_t>() : nullptr,
            N, epsilon);
    });
}

}  // extern "C"
