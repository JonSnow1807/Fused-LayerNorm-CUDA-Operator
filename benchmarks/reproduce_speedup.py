"""
Reproduce the claimed 1.85-2.57x speedup
Run: python benchmarks/reproduce_speedup.py
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda
import numpy as np

print("="*60)
print("LAYERNORM SPEEDUP REPRODUCTION")
print("="*60)

def benchmark(name, batch, hidden, iterations=1000):
    x = torch.randn(batch, hidden, device='cuda')
    ln = nn.LayerNorm(hidden).cuda()
    
    # Warmup
    for _ in range(200):
        _ = ln(x)
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    
    torch.cuda.synchronize()
    
    # Time PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _ = ln(x)
    end.record()
    torch.cuda.synchronize()
    pt_time = start.elapsed_time(end) / iterations
    
    # Time Ours  
    start.record()
    for _ in range(iterations):
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    end.record()
    torch.cuda.synchronize()
    our_time = start.elapsed_time(end) / iterations
    
    speedup = pt_time / our_time
    print(f"{name:20} PyTorch: {pt_time:.4f}ms, Ours: {our_time:.4f}ms, Speedup: {speedup:.2f}x")
    return speedup

print("Building module...")
speedups = []
speedups.append(benchmark("BERT (32×768)", 32, 768))
speedups.append(benchmark("GPT-2 (32×1024)", 32, 1024))
speedups.append(benchmark("GPT-3 (32×4096)", 32, 4096))
speedups.append(benchmark("Large (64×4096)", 64, 4096))
speedups.append(benchmark("Odd (17×1023)", 17, 1023))

print("="*60)
print(f"Average Speedup: {np.mean(speedups):.2f}x")
print("Note: Results vary based on cache utilization")
print("Expected: 1.85x (realistic) to 2.57x (optimal)")
print("="*60)
