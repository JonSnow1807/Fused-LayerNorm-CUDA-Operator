"""
Functional interface for Fused LayerNorm operations.

This module provides additional utilities and functional interfaces
for the fused LayerNorm implementation.
"""

import torch
from typing import Optional, Tuple
from .layernorm import FusedLayerNormFunction


def check_input_dtype_and_device(input: torch.Tensor, 
                                weight: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None) -> None:
    """
    Validate input tensor properties for CUDA operations.
    
    Args:
        input: Input tensor to validate
        weight: Optional weight tensor
        bias: Optional bias tensor
        
    Raises:
        RuntimeError: If tensors are not on CUDA or have mismatched properties
    """
    if not input.is_cuda:
        raise RuntimeError("FusedLayerNorm requires CUDA tensors")
    
    if weight is not None:
        if not weight.is_cuda:
            raise RuntimeError("Weight must be a CUDA tensor")
        if weight.device != input.device:
            raise RuntimeError("Weight and input must be on the same device")
        if weight.dtype != input.dtype:
            raise RuntimeError("Weight and input must have the same dtype")
    
    if bias is not None:
        if not bias.is_cuda:
            raise RuntimeError("Bias must be a CUDA tensor")
        if bias.device != input.device:
            raise RuntimeError("Bias and input must be on the same device")
        if bias.dtype != input.dtype:
            raise RuntimeError("Bias and input must have the same dtype")


def get_optimal_config(batch_size: int, 
                      hidden_size: int, 
                      dtype: torch.dtype = torch.float32) -> dict:
    """
    Get optimal configuration parameters for given input dimensions.
    
    Args:
        batch_size: Batch dimension
        hidden_size: Hidden dimension
        dtype: Data type of tensors
        
    Returns:
        Dictionary with optimal configuration parameters
    """
    config = {
        'use_mixed_precision': dtype == torch.float16,
        'threads_per_block': min(hidden_size, 1024),
        'enable_persistent_kernel': batch_size >= 16 and hidden_size <= 2048,
        'use_vectorized_load': hidden_size % 4 == 0,
    }
    
    # Adjust for specific architectures
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere and newer
        config['use_tensor_cores'] = dtype == torch.float16 and hidden_size % 8 == 0
    else:
        config['use_tensor_cores'] = False
    
    return config


def benchmark_config(input_shape: Tuple[int, ...], 
                    num_iterations: int = 100,
                    warmup_iterations: int = 10) -> dict:
    """
    Benchmark the fused LayerNorm implementation with given configuration.
    
    Args:
        input_shape: Shape of input tensor
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    device = torch.cuda.current_device()
    dtype = torch.float32
    
    # Create test tensors
    input_tensor = torch.randn(input_shape, device=device, dtype=dtype)
    normalized_shape = (input_shape[-1],)
    weight = torch.ones(normalized_shape, device=device, dtype=dtype)
    bias = torch.zeros(normalized_shape, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = FusedLayerNormFunction.apply(input_tensor, normalized_shape, weight, bias, 1e-5)
    
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        output = FusedLayerNormFunction.apply(input_tensor, normalized_shape, weight, bias, 1e-5)
    torch.cuda.synchronize()
    forward_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    # Benchmark backward pass
    grad_output = torch.randn_like(output)
    start_time = time.time()
    for _ in range(num_iterations):
        output = FusedLayerNormFunction.apply(input_tensor, normalized_shape, weight, bias, 1e-5)
        output.backward(grad_output)
    torch.cuda.synchronize()
    backward_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    return {
        'forward_time_ms': forward_time,
        'backward_time_ms': backward_time,
        'total_time_ms': forward_time + backward_time,
        'throughput_gb_s': (input_tensor.numel() * input_tensor.element_size() * 2) / (forward_time * 1e6)
    }


class LayerNormProfiler:
    """
    Profiler for analyzing LayerNorm performance characteristics.
    """
    
    def __init__(self):
        self.profiles = []
        self.enabled = False
    
    def start(self):
        """Start profiling."""
        self.enabled = True
        self.profiles = []
    
    def stop(self):
        """Stop profiling and return results."""
        self.enabled = False
        return self.analyze_profiles()
    
    def record(self, batch_size: int, hidden_size: int, 
               forward_time: float, backward_time: float):
        """Record a profile entry."""
        if self.enabled:
            self.profiles.append({
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'forward_time_ms': forward_time,
                'backward_time_ms': backward_time,
                'total_time_ms': forward_time + backward_time
            })
    
    def analyze_profiles(self) -> dict:
        """Analyze collected profiles."""
        if not self.profiles:
            return {}
        
        import statistics
        
        forward_times = [p['forward_time_ms'] for p in self.profiles]
        backward_times = [p['backward_time_ms'] for p in self.profiles]
        total_times = [p['total_time_ms'] for p in self.profiles]
        
        return {
            'num_calls': len(self.profiles),
            'forward': {
                'mean_ms': statistics.mean(forward_times),
                'std_ms': statistics.stdev(forward_times) if len(forward_times) > 1 else 0,
                'min_ms': min(forward_times),
                'max_ms': max(forward_times)
            },
            'backward': {
                'mean_ms': statistics.mean(backward_times),
                'std_ms': statistics.stdev(backward_times) if len(backward_times) > 1 else 0,
                'min_ms': min(backward_times),
                'max_ms': max(backward_times)
            },
            'total': {
                'mean_ms': statistics.mean(total_times),
                'std_ms': statistics.stdev(total_times) if len(total_times) > 1 else 0,
                'min_ms': min(total_times),
                'max_ms': max(total_times)
            }
        }


# Global profiler instance
_profiler = LayerNormProfiler()


def enable_profiling():
    """Enable global profiling of LayerNorm operations."""
    _profiler.start()


def disable_profiling():
    """Disable profiling and return results."""
    return _profiler.stop()


def get_profiler():
    """Get the global profiler instance."""
    return _profiler