"""
Edge case testing for publication claims
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda

print("="*80)
print("EDGE CASE TESTING")
print("="*80)

edge_cases = [
    (1, 1, "Minimum size"),
    (1, 17, "Prime dimension"),
    (1000, 1, "Many batch, size 1"),
    (1, 100000, "Huge hidden dim"),
    (512, 512, "Square"),
    (1, 32767, "Near int16 max"),
    (1, 65536, "Power of 2 large"),
    (13, 13, "Small prime square"),
    (1, 4095, "4096-1"),
    (1, 4097, "4096+1"),
]

print(f"{'Case':30} {'Status':>10} {'Speedup':>10} {'Error':>12}")
print("-" * 70)

for batch, hidden, description in edge_cases:
    try:
        x = torch.randn(batch, hidden, device='cuda')
        ln = nn.LayerNorm(hidden).cuda()
        
        # Check correctness
        pytorch_out = ln(x)
        our_out = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        
        correct = torch.allclose(pytorch_out, our_out, rtol=1e-4, atol=1e-5)
        max_error = (pytorch_out - our_out).abs().max().item()
        
        # Quick performance check
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        pt_time = start.elapsed_time(end)
        
        start.record()
        for _ in range(100):
            _ = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
        end.record()
        torch.cuda.synchronize()
        our_time = start.elapsed_time(end)
        
        speedup = pt_time / our_time
        status = "✅" if correct else "❌"
        
        print(f"{description:30} {status:>10} {speedup:9.2f}x {max_error:12.2e}")
        
    except Exception as e:
        print(f"{description:30} {'❌ ERROR':>10} {str(e)[:40]}")

print("="*80)
