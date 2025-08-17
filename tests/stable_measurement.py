"""
STABLE measurement methodology for publication
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda
import numpy as np

print("="*70)
print("STABLE PUBLICATION-READY MEASUREMENTS")
print("="*70)

def stable_benchmark(batch, hidden, warmup=1000, iterations=1000):
    """
    Proper benchmarking with extensive warmup
    """
    x = torch.randn(batch, hidden, device='cuda')
    ln = nn.LayerNorm(hidden).cuda()
    
    print(f"\nConfig: {batch}×{hidden}")
    print("Warming up GPU (this eliminates compilation/cold start)...")
    
    # EXTENSIVE warmup to eliminate ALL startup effects
    for _ in range(warmup):
        _ = ln(x)
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    
    torch.cuda.synchronize()
    print("Warmup complete. Taking measurements...")
    
    # Now take multiple measurements
    pt_times = []
    our_times = []
    
    for i in range(10):
        # PyTorch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        pt_times.append(start.elapsed_time(end) / iterations)
        
        # Ours
        start.record()
        for _ in range(iterations):
            _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        end.record()
        torch.cuda.synchronize()
        our_times.append(start.elapsed_time(end) / iterations)
    
    # Remove outliers (first measurement often bad)
    pt_times = pt_times[1:]  # Skip first
    our_times = our_times[1:]
    
    pt_mean = np.mean(pt_times)
    pt_std = np.std(pt_times)
    our_mean = np.mean(our_times)
    our_std = np.std(our_times)
    
    speedup_mean = pt_mean / our_mean
    
    # Calculate coefficient of variation (should be <5% for stable)
    pt_cv = (pt_std / pt_mean) * 100
    our_cv = (our_std / our_mean) * 100
    
    print(f"PyTorch: {pt_mean:.4f} ± {pt_std:.4f} ms (CV: {pt_cv:.1f}%)")
    print(f"Ours:    {our_mean:.4f} ± {our_std:.4f} ms (CV: {our_cv:.1f}%)")
    print(f"Speedup: {speedup_mean:.2f}x")
    print(f"Stability: {'✅ STABLE' if max(pt_cv, our_cv) < 5 else '⚠️ UNSTABLE'}")
    
    return speedup_mean, max(pt_cv, our_cv)

# Test multiple configurations
configs = [
    (32, 768, "BERT"),
    (32, 4096, "GPT-3"),
    (17, 1023, "Odd"),
]

speedups = []
stabilities = []

for batch, hidden, name in configs:
    speedup, cv = stable_benchmark(batch, hidden)
    speedups.append(speedup)
    stabilities.append(cv)

print("\n" + "="*70)
print("PUBLICATION-READY SUMMARY")
print("="*70)
print(f"Average Speedup: {np.mean(speedups):.2f}x")
print(f"Speedup Range: {min(speedups):.2f}x - {max(speedups):.2f}x")
print(f"Maximum CV: {max(stabilities):.1f}%")
print(f"All Stable (CV<5%): {'✅ YES' if all(s < 5 for s in stabilities) else '❌ NO'}")

print("\nFor publication, report:")
print(f"'After extensive warmup to eliminate startup effects, we observe")
print(f" {np.mean(speedups):.2f}x average speedup with coefficient of variation < {max(stabilities):.1f}%'")
print("="*70)
