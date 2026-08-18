"""
Publication-ready benchmark with honest methodology
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda
import numpy as np
import json

print("="*80)
print("PUBLICATION-READY BENCHMARK")
print("="*80)

def benchmark_realistic(batch, hidden, samples=100):
    """Realistic benchmark with different tensors each iteration"""
    ln = nn.LayerNorm(hidden).cuda()
    
    pt_times = []
    our_times = []
    
    for _ in range(samples):
        # Fresh tensor each iteration (realistic scenario)
        x = torch.randn(batch, hidden, device='cuda')
        
        # PyTorch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        pt_times.append(start.elapsed_time(end))
        
        # Ours
        start.record()
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        end.record()
        torch.cuda.synchronize()
        our_times.append(start.elapsed_time(end))
    
    return {
        'pytorch_mean': np.mean(pt_times),
        'pytorch_std': np.std(pt_times),
        'our_mean': np.mean(our_times),
        'our_std': np.std(our_times),
        'speedup': np.mean(pt_times) / np.mean(our_times)
    }

def benchmark_cached(batch, hidden, iterations=1000):
    """Best-case benchmark with tensor reuse (cache-friendly)"""
    x = torch.randn(batch, hidden, device='cuda')
    ln = nn.LayerNorm(hidden).cuda()
    
    # Warmup
    for _ in range(200):
        _ = ln(x)
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    
    torch.cuda.synchronize()
    
    # PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _ = ln(x)
    end.record()
    torch.cuda.synchronize()
    pt_time = start.elapsed_time(end) / iterations
    
    # Ours
    start.record()
    for _ in range(iterations):
        _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
    end.record()
    torch.cuda.synchronize()
    our_time = start.elapsed_time(end) / iterations
    
    return {
        'pytorch_time': pt_time,
        'our_time': our_time,
        'speedup': pt_time / our_time
    }

configs = [
    (32, 768, "BERT"),
    (32, 1024, "GPT-2"),
    (32, 4096, "GPT-3"),
    (64, 4096, "Large Batch"),
    (17, 1023, "Odd Dimensions"),
]

print("\n" + "="*80)
print("REALISTIC PERFORMANCE (Different Tensors):")
print("-"*80)
print(f"{'Config':20} {'PyTorch(ms)':>12} {'Ours(ms)':>10} {'Speedup':>10}")
print("-"*55)

realistic_results = []
for batch, hidden, name in configs:
    result = benchmark_realistic(batch, hidden)
    realistic_results.append(result)
    print(f"{name:20} {result['pytorch_mean']:12.4f} {result['our_mean']:10.4f} {result['speedup']:10.2f}x")

avg_realistic = np.mean([r['speedup'] for r in realistic_results])

print("\n" + "="*80)
print("BEST-CASE PERFORMANCE (Cached/Same Tensor):")
print("-"*80)
print(f"{'Config':20} {'PyTorch(ms)':>12} {'Ours(ms)':>10} {'Speedup':>10}")
print("-"*55)

cached_results = []
for batch, hidden, name in configs:
    result = benchmark_cached(batch, hidden)
    cached_results.append(result)
    print(f"{name:20} {result['pytorch_time']:12.4f} {result['our_time']:10.4f} {result['speedup']:10.2f}x")

avg_cached = np.mean([r['speedup'] for r in cached_results])

print("\n" + "="*80)
print("PUBLICATION-READY CLAIMS:")
print("="*80)
print(f"Realistic Speedup (different tensors): {avg_realistic:.2f}x")
print(f"Best-Case Speedup (cached tensors):    {avg_cached:.2f}x")
print("\nHonest claim for paper:")
print(f'  "We achieve {avg_realistic:.1f}x speedup in realistic scenarios')
print(f'   and up to {avg_cached:.1f}x speedup with optimal cache utilization."')
print("="*80)

# Save results
results = {
    'realistic': realistic_results,
    'cached': cached_results,
    'summary': {
        'realistic_speedup': avg_realistic,
        'cached_speedup': avg_cached
    }
}

with open('publication_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print("\n✅ Results saved to publication_results.json")
