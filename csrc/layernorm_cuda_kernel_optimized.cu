/*
 * Optimized Fused LayerNorm CUDA Kernel
 * Target: 1.4x speedup over PyTorch
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

namespace cg = cooperative_groups;

// Optimized warp reduction using __shfl_down_sync
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized block reduction
template<typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared, int tid, int blockSize) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (tid < blockSize / WARP_SIZE) {
        val = shared[tid];
    } else {
        val = 0;
    }
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Optimized forward kernel with aggressive unrolling
template<typename T, typename T_ACC>
__global__ void layernorm_forward_kernel_optimized(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T_ACC* __restrict__ mean,
    T_ACC* __restrict__ rstd,
    const int batch_size,
    const int hidden_size,
    const T_ACC epsilon) {
    
    extern __shared__ char shared_data[];
    T_ACC* s_mean = reinterpret_cast<T_ACC*>(shared_data);
    T_ACC* s_var = s_mean + blockDim.x / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    const T* input_ptr = input + bid * hidden_size;
    T* output_ptr = output + bid * hidden_size;
    
    // Use float4 for vectorized loads when possible
    const int vec_hidden_size = hidden_size / 4;
    const float4* input_ptr_vec = reinterpret_cast<const float4*>(input_ptr);
    float4* output_ptr_vec = reinterpret_cast<float4*>(output_ptr);
    
    // First pass: compute mean and variance
    T_ACC local_sum = 0;
    T_ACC local_sum_sq = 0;
    
    // Vectorized loop for hidden_size divisible by 4
    if (hidden_size % 4 == 0) {
        #pragma unroll 8
        for (int i = tid; i < vec_hidden_size; i += blockDim.x) {
            float4 vec = input_ptr_vec[i];
            float vals[4] = {vec.x, vec.y, vec.z, vec.w};
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                T_ACC val = vals[j];
                local_sum += val;
                local_sum_sq += val * val;
            }
        }
    } else {
        #pragma unroll 8
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            T_ACC val = static_cast<T_ACC>(input_ptr[i]);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Reduce across block
    T_ACC block_sum = blockReduceSum(local_sum, s_mean, tid, blockDim.x);
    T_ACC block_sum_sq = blockReduceSum(local_sum_sq, s_var, tid, blockDim.x);
    
    // Compute statistics
    if (tid == 0) {
        T_ACC sample_mean = block_sum / hidden_size;
        T_ACC sample_var = (block_sum_sq / hidden_size) - (sample_mean * sample_mean);
        T_ACC sample_rstd = rsqrtf(sample_var + epsilon);
        
        if (mean != nullptr) mean[bid] = sample_mean;
        if (rstd != nullptr) rstd[bid] = sample_rstd;
        
        s_mean[0] = sample_mean;
        s_var[0] = sample_rstd;
    }
    __syncthreads();
    
    T_ACC sample_mean = s_mean[0];
    T_ACC sample_rstd = s_var[0];
    
    // Second pass: normalize and apply affine transformation
    if (gamma != nullptr && beta != nullptr) {
        if (hidden_size % 4 == 0) {
            const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
            const float4* beta_vec = reinterpret_cast<const float4*>(beta);
            
            #pragma unroll 8
            for (int i = tid; i < vec_hidden_size; i += blockDim.x) {
                float4 input_vec = input_ptr_vec[i];
                float4 gamma_vec_val = gamma_vec[i];
                float4 beta_vec_val = beta_vec[i];
                
                float4 output_vec;
                output_vec.x = ((input_vec.x - sample_mean) * sample_rstd) * gamma_vec_val.x + beta_vec_val.x;
                output_vec.y = ((input_vec.y - sample_mean) * sample_rstd) * gamma_vec_val.y + beta_vec_val.y;
                output_vec.z = ((input_vec.z - sample_mean) * sample_rstd) * gamma_vec_val.z + beta_vec_val.z;
                output_vec.w = ((input_vec.w - sample_mean) * sample_rstd) * gamma_vec_val.w + beta_vec_val.w;
                
                output_ptr_vec[i] = output_vec;
            }
        } else {
            #pragma unroll 8
            for (int i = tid; i < hidden_size; i += blockDim.x) {
                T_ACC val = static_cast<T_ACC>(input_ptr[i]);
                T_ACC normalized = (val - sample_mean) * sample_rstd;
                normalized = normalized * static_cast<T_ACC>(gamma[i]) + static_cast<T_ACC>(beta[i]);
                output_ptr[i] = static_cast<T>(normalized);
            }
        }
    } else {
        #pragma unroll 8
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            T_ACC val = static_cast<T_ACC>(input_ptr[i]);
            T_ACC normalized = (val - sample_mean) * sample_rstd;
            output_ptr[i] = static_cast<T>(normalized);
        }
    }
}

// Kernel launcher with optimized configuration
void layernorm_forward_cuda_optimized(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor output,
    at::Tensor mean,
    at::Tensor rstd,
    float epsilon) {
    
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);
    
    // Optimized thread configuration
    int threads = 256;  // Fixed for better occupancy
    if (hidden_size >= 4096) {
        threads = 512;
    }
    if (hidden_size >= 8192) {
        threads = 1024;
    }
    
    const int blocks = batch_size;
    const int shared_mem_size = 2 * (threads / WARP_SIZE) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layernorm_forward_cuda_optimized", [&] {
        using T_ACC = float;  // Always use float for accuracy
        
        layernorm_forward_kernel_optimized<scalar_t, T_ACC><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            beta.defined() ? beta.data_ptr<scalar_t>() : nullptr,
            mean.data_ptr<T_ACC>(),
            rstd.data_ptr<T_ACC>(),
            batch_size,
            hidden_size,
            static_cast<T_ACC>(epsilon)
        );
    });
    
    AT_CUDA_CHECK(cudaGetLastError());
}