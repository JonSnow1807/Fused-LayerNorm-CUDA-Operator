"""
Statistical validation for publication-quality claims
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda
import numpy as np
from scipy import stats
import json

def benchmark_statistical(batch, hidden, samples=30, iterations=1000):
    """Get statistically significant timing with confidence intervals"""
    
    x = torch.randn(batch, hidden, device='cuda')
    ln = nn.LayerNorm(hidden).cuda()
    
    pytorch_times = []
    our_times = []
    
    for _ in range(samples):
        # Fresh tensors each run to avoid cache effects
        x = torch.randn(batch, hidden, device='cuda')
        
        # Warmup
        for _ in range(100):
            _ = ln(x)
            _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        
        torch.cuda.synchronize()
        
        # PyTorch timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        pytorch_times.append(start.elapsed_time(end) / iterations)
        
        # Our timing
        start.record()
        for _ in range(iterations):
            _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        end.record()
        torch.cuda.synchronize()
        our_times.append(start.elapsed_time(end) / iterations)
    
    # Statistical analysis
    pt_mean, pt_std = np.mean(pytorch_times), np.std(pytorch_times)
    our_mean, our_std = np.mean(our_times), np.std(our_times)
    
    # T-test for significance
    t_stat, p_value = stats.ttest_ind(pytorch_times, our_times)
    
    # Confidence intervals (95%)
    pt_ci = stats.t.interval(0.95, len(pytorch_times)-1, loc=pt_mean, scale=pt_std/np.sqrt(len(pytorch_times)))
    our_ci = stats.t.interval(0.95, len(our_times)-1, loc=our_mean, scale=our_std/np.sqrt(len(our_times)))
    
    speedup = pt_mean / our_mean
    speedup_std = speedup * np.sqrt((pt_std/pt_mean)**2 + (our_std/our_mean)**2)
    
    return {
        'config': f'{batch}×{hidden}',
        'pytorch_mean': pt_mean,
        'pytorch_std': pt_std,
        'pytorch_ci': pt_ci,
        'our_mean': our_mean,
        'our_std': our_std,
        'our_ci': our_ci,
        'speedup': speedup,
        'speedup_std': speedup_std,
        'p_value': p_value,
        'significant': p_value < 0.001
    }

# Test multiple configurations
configs = [
    (1, 768),
    (8, 768),
    (16, 768),
    (32, 768),
    (32, 1024),
    (32, 2048),
    (32, 4096),
    (64, 4096),
    (128, 4096),
    # Edge cases
    (17, 1023),  # Both odd
    (1, 1),      # Minimum
    (200, 10000), # Large
]

print("="*80)
print("STATISTICAL VALIDATION FOR PUBLICATION")
print("="*80)
print("\nRunning 30 samples per configuration for statistical significance...")
print("This will take a few minutes...\n")

results = []
for batch, hidden in configs:
    print(f"Testing {batch}×{hidden}...", end='', flush=True)
    try:
        result = benchmark_statistical(batch, hidden)
        results.append(result)
        print(f" Speedup: {result['speedup']:.2f}x ± {result['speedup_std']:.2f} (p={result['p_value']:.4f})")
    except Exception as e:
        print(f" Error: {e}")

# Summary statistics
significant_results = [r for r in results if r['significant']]
print("\n" + "="*80)
print("SUMMARY:")
print(f"Configurations tested: {len(results)}")
print(f"Statistically significant (p<0.001): {len(significant_results)}")
print(f"Average speedup: {np.mean([r['speedup'] for r in results]):.2f}x")
print(f"Minimum speedup: {np.min([r['speedup'] for r in results]):.2f}x")
print(f"Maximum speedup: {np.max([r['speedup'] for r in results]):.2f}x")

# Save for paper
with open('statistical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to statistical_results.json")
print("="*80)
