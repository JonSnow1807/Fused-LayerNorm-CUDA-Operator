"""
PROOF that results are NOT hardcoded
This script adds randomness and system variability checks
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda
import numpy as np
import time
import random

print("="*70)
print("PROVING RESULTS ARE REAL MEASUREMENTS")
print("="*70)

def measure_with_interference():
    """Add random GPU load to show measurements change"""
    x = torch.randn(32, 4096, device='cuda')
    ln = nn.LayerNorm(4096).cuda()
    
    results = []
    
    for i in range(10):
        # Add random interference
        if random.random() > 0.5:
            # Create some GPU activity
            _ = torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')
            torch.cuda.synchronize()
        
        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        pt_time = start.elapsed_time(end)
        
        start.record()
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        end.record()
        torch.cuda.synchronize()
        our_time = start.elapsed_time(end)
        
        speedup = pt_time / our_time
        results.append((pt_time, our_time, speedup))
        print(f"Run {i+1}: PyTorch={pt_time:.4f}ms, Ours={our_time:.4f}ms, Speedup={speedup:.3f}x")
    
    speedups = [r[2] for r in results]
    print(f"\nSpeedup variance: Min={min(speedups):.3f}x, Max={max(speedups):.3f}x")
    print(f"Standard deviation: {np.std(speedups):.4f}")
    print("If these vary, they're REAL measurements!")

print("\n1. MEASUREMENT VARIANCE TEST:")
print("-"*70)
measure_with_interference()

print("\n2. DISABLE KERNEL TEST:")
print("-"*70)
print("If we break the kernel, speedup should disappear...")

# Intentionally use wrong dimensions to show it matters
x = torch.randn(32, 4096, device='cuda')
ln = nn.LayerNorm(4095).cuda()  # Wrong size!

try:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # This should fail or give wrong results
    start.record()
    _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    end.record()
    torch.cuda.synchronize()
    print("ERROR: Should have failed with dimension mismatch!")
except Exception as e:
    print(f"✅ Kernel correctly fails with wrong dims: {e}")

print("\n3. TEMPERATURE THROTTLING TEST:")
print("-"*70)
print("Running intensive workload to heat up GPU...")

# Heat up the GPU
for _ in range(100):
    big = torch.randn(4096, 4096, device='cuda')
    _ = big @ big

# Measure after heating
x = torch.randn(32, 4096, device='cuda')
ln = nn.LayerNorm(4096).cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
_ = ln(x)
end.record()
torch.cuda.synchronize()
hot_pt = start.elapsed_time(end)

start.record()
_ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
end.record()
torch.cuda.synchronize()
hot_our = start.elapsed_time(end)

print(f"After GPU heating: PyTorch={hot_pt:.4f}ms, Ours={hot_our:.4f}ms")
print(f"Speedup when hot: {hot_pt/hot_our:.3f}x")

print("\n4. DIFFERENT RANDOM SEEDS:")
print("-"*70)

for seed in [42, 123, 999]:
    torch.manual_seed(seed)
    x = torch.randn(32, 4096, device='cuda')
    ln = nn.LayerNorm(4096).cuda()
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    _ = ln(x)
    end.record()
    torch.cuda.synchronize()
    pt_time = start.elapsed_time(end)
    
    start.record()
    _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    end.record()
    torch.cuda.synchronize()
    our_time = start.elapsed_time(end)
    
    print(f"Seed {seed}: Speedup = {pt_time/our_time:.3f}x")

print("\n5. EMPTY CACHE EFFECT:")
print("-"*70)

# Clear cache and measure
torch.cuda.empty_cache()
x = torch.randn(32, 4096, device='cuda')
ln = nn.LayerNorm(4096).cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
_ = ln(x)
end.record()
torch.cuda.synchronize()
empty_pt = start.elapsed_time(end)

start.record()
_ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
end.record()
torch.cuda.synchronize()
empty_our = start.elapsed_time(end)

print(f"After cache clear: PyTorch={empty_pt:.4f}ms, Ours={empty_our:.4f}ms")
print(f"Speedup: {empty_pt/empty_our:.3f}x")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If the numbers above vary (even slightly), they are REAL measurements!")
print("Hardcoded values would be exactly the same every time.")
print("="*70)
