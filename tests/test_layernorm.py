"""
Comprehensive unit tests for Fused LayerNorm CUDA implementation.

Tests correctness, numerical accuracy, gradient computation, and edge cases.
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Tuple
import itertools

from fused_layernorm import FusedLayerNorm, fused_layer_norm


class TestFusedLayerNorm:
    """Test suite for FusedLayerNorm implementation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Default test parameters
        self.device = 'cuda'
        self.rtol = 1e-5
        self.atol = 1e-5
        self.grad_rtol = 1e-4
        self.grad_atol = 1e-4
    
    @pytest.mark.parametrize("batch_size,hidden_size,dtype", [
        (1, 768, torch.float32),
        (8, 768, torch.float32),
        (32, 1024, torch.float32),
        (64, 2048, torch.float32),
        (128, 4096, torch.float32),
        (1, 768, torch.float16),
        (32, 1024, torch.float16),
        (64, 2048, torch.float16),
    ])
    def test_forward_correctness(self, batch_size: int, hidden_size: int, dtype: torch.dtype):
        """Test forward pass correctness against PyTorch implementation."""
        # Create input
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device, dtype=dtype)
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size, dtype=dtype).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size, dtype=dtype).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)
        
        # Forward pass
        pytorch_output = pytorch_ln(input_tensor)
        fused_output = fused_ln(input_tensor)
        
        # Check correctness
        torch.testing.assert_close(
            fused_output, pytorch_output,
            rtol=self.rtol if dtype == torch.float32 else 1e-3,
            atol=self.atol if dtype == torch.float32 else 1e-3
        )
    
    @pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
        (2, 128, 768),
        (4, 256, 1024),
        (8, 512, 2048),
        (1, 1024, 4096),
    ])
    def test_3d_input(self, batch_size: int, seq_len: int, hidden_size: int):
        """Test with 3D input (batch, seq_len, hidden_size)."""
        # Create 3D input
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, 
                                 device=self.device, dtype=torch.float32)
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)
        
        # Forward pass
        pytorch_output = pytorch_ln(input_tensor)
        fused_output = fused_ln(input_tensor)
        
        # Check correctness
        torch.testing.assert_close(
            fused_output, pytorch_output,
            rtol=self.rtol, atol=self.atol
        )
    
    @pytest.mark.parametrize("hidden_size,elementwise_affine", [
        (768, True),
        (768, False),
        (1024, True),
        (1024, False),
    ])
    def test_no_affine_parameters(self, hidden_size: int, elementwise_affine: bool):
        """Test LayerNorm without affine parameters."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size, elementwise_affine=elementwise_affine).to(self.device)
        
        # Forward pass
        pytorch_output = pytorch_ln(input_tensor)
        fused_output = fused_ln(input_tensor)
        
        # Check correctness
        torch.testing.assert_close(
            fused_output, pytorch_output,
            rtol=self.rtol, atol=self.atol
        )
    
    @pytest.mark.parametrize("batch_size,hidden_size", [
        (16, 768),
        (32, 1024),
        (64, 2048),
    ])
    def test_backward_correctness(self, batch_size: int, hidden_size: int):
        """Test backward pass gradient computation."""
        # Create input with gradient
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device, 
                                 requires_grad=True)
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)
        
        # Forward pass
        input_pytorch = input_tensor.clone().detach().requires_grad_(True)
        input_fused = input_tensor.clone().detach().requires_grad_(True)
        
        pytorch_output = pytorch_ln(input_pytorch)
        fused_output = fused_ln(input_fused)
        
        # Create gradient
        grad_output = torch.randn_like(pytorch_output)
        
        # Backward pass
        pytorch_output.backward(grad_output)
        fused_output.backward(grad_output)
        
        # Check input gradients
        torch.testing.assert_close(
            input_fused.grad, input_pytorch.grad,
            rtol=self.grad_rtol, atol=self.grad_atol
        )
        
        # Check weight gradients
        torch.testing.assert_close(
            fused_ln.weight.grad, pytorch_ln.weight.grad,
            rtol=self.grad_rtol, atol=self.grad_atol
        )
        
        # Check bias gradients
        torch.testing.assert_close(
            fused_ln.bias.grad, pytorch_ln.bias.grad,
            rtol=self.grad_rtol, atol=self.grad_atol
        )
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation across multiple backward passes."""
        batch_size, hidden_size = 32, 768
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)
        
        # Multiple forward/backward passes
        for _ in range(3):
            input_tensor = torch.randn(batch_size, hidden_size, device=self.device, 
                                     requires_grad=True)
            
            pytorch_output = pytorch_ln(input_tensor)
            fused_output = fused_ln(input_tensor.clone())
            
            loss_pytorch = pytorch_output.sum()
            loss_fused = fused_output.sum()
            
            loss_pytorch.backward()
            loss_fused.backward()
        
        # Check accumulated gradients
        torch.testing.assert_close(
            fused_ln.weight.grad, pytorch_ln.weight.grad,
            rtol=self.grad_rtol, atol=self.grad_atol
        )
    
    @pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-7, 1e-8])
    def test_different_epsilon(self, epsilon: float):
        """Test with different epsilon values."""
        batch_size, hidden_size = 32, 768
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        # Create layers with different epsilon
        pytorch_ln = nn.LayerNorm(hidden_size, eps=epsilon).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size, eps=epsilon).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)
        
        # Forward pass
        pytorch_output = pytorch_ln(input_tensor)
        fused_output = fused_ln(input_tensor)
        
        # Check correctness
        torch.testing.assert_close(
            fused_output, pytorch_output,
            rtol=self.rtol, atol=self.atol
        )
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch_size, hidden_size = 16, 768
        
        # Test cases with extreme values
        test_cases = [
            torch.ones(batch_size, hidden_size, device=self.device) * 1e-8,  # Very small
            torch.ones(batch_size, hidden_size, device=self.device) * 1e8,   # Very large
            torch.randn(batch_size, hidden_size, device=self.device) * 1e-4, # Small variance
            torch.randn(batch_size, hidden_size, device=self.device) * 1e4,  # Large variance
        ]
        
        for input_tensor in test_cases:
            pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
            fused_ln = FusedLayerNorm(hidden_size).to(self.device)
            
            # Copy weights
            with torch.no_grad():
                fused_ln.weight.copy_(pytorch_ln.weight)
                fused_ln.bias.copy_(pytorch_ln.bias)
            
            # Forward pass
            pytorch_output = pytorch_ln(input_tensor)
            fused_output = fused_ln(input_tensor)
            
            # Check outputs are finite
            assert torch.isfinite(fused_output).all(), "Fused output contains non-finite values"
            
            # Check correctness (with relaxed tolerance for extreme cases)
            torch.testing.assert_close(
                fused_output, pytorch_output,
                rtol=1e-3, atol=1e-3
            )
    
    def test_memory_efficiency(self):
        """Test memory usage is reduced compared to PyTorch."""
        batch_size, hidden_size = 64, 4096
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        # Measure PyTorch memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
        start_mem = torch.cuda.memory_allocated()
        _ = pytorch_ln(input_tensor.clone())
        pytorch_mem = torch.cuda.max_memory_allocated() - start_mem
        
        # Measure Fused memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        start_mem = torch.cuda.memory_allocated()
        _ = fused_ln(input_tensor.clone())
        fused_mem = torch.cuda.max_memory_allocated() - start_mem
        
        # Check memory reduction
        memory_reduction = 1 - (fused_mem / pytorch_mem)
        assert memory_reduction > 0.1, f"Expected memory reduction, got {memory_reduction:.2%}"
    
    def test_functional_interface(self):
        """Test functional interface."""
        batch_size, hidden_size = 32, 768
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        weight = torch.ones(hidden_size, device=self.device)
        bias = torch.zeros(hidden_size, device=self.device)
        
        # Compare with PyTorch functional
        pytorch_output = torch.nn.functional.layer_norm(
            input_tensor, (hidden_size,), weight, bias
        )
        fused_output = fused_layer_norm(
            input_tensor, (hidden_size,), weight, bias
        )
        
        torch.testing.assert_close(
            fused_output, pytorch_output,
            rtol=self.rtol, atol=self.atol
        )
    
    @pytest.mark.parametrize("seed", range(5))
    def test_deterministic(self, seed: int):
        """Test that results are deterministic."""
        torch.manual_seed(seed)
        
        batch_size, hidden_size = 32, 768
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        
        # Run multiple times
        outputs = []
        for _ in range(3):
            output = fused_ln(input_tensor.clone())
            outputs.append(output)
        
        # Check all outputs are identical
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i], rtol=0, atol=0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single element batch
        input_tensor = torch.randn(1, 768, device=self.device)
        fused_ln = FusedLayerNorm(768).to(self.device)
        output = fused_ln(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Hidden size of 1
        input_tensor = torch.randn(32, 1, device=self.device)
        fused_ln = FusedLayerNorm(1).to(self.device)
        output = fused_ln(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Very large hidden size
        input_tensor = torch.randn(2, 16384, device=self.device)
        fused_ln = FusedLayerNorm(16384).to(self.device)
        output = fused_ln(input_tensor)
        assert output.shape == input_tensor.shape


@pytest.mark.benchmark
class TestPerformance:
    """Performance regression tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up performance tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        self.device = 'cuda'
        self.min_speedup = 1.2  # Minimum acceptable speedup
    
    @pytest.mark.parametrize("batch_size,hidden_size", [
        (32, 768),   # BERT-Base
        (32, 1024),  # BERT-Large
        (16, 2048),  # GPT-2 Large
        (8, 4096),   # GPT-3 Large
    ])
    def test_performance_regression(self, batch_size: int, hidden_size: int):
        """Test that performance meets minimum requirements."""
        import time
        
        input_tensor = torch.randn(batch_size, hidden_size, device=self.device)
        
        # Create layers
        pytorch_ln = nn.LayerNorm(hidden_size).to(self.device)
        fused_ln = FusedLayerNorm(hidden_size).to(self.device)
        
        # Warmup
        for _ in range(50):
            _ = pytorch_ln(input_tensor)
            _ = fused_ln(input_tensor)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(100):
            _ = pytorch_ln(input_tensor)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # Benchmark Fused
        start = time.time()
        for _ in range(100):
            _ = fused_ln(input_tensor)
        torch.cuda.synchronize()
        fused_time = time.time() - start
        
        speedup = pytorch_time / fused_time
        assert speedup >= self.min_speedup, \
            f"Performance regression: {speedup:.2f}x < {self.min_speedup}x"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])