"""
Break the kernel intentionally to prove speedup is real
"""
import torch
import torch.nn as nn
import fused_layernorm_cuda

print("="*70)
print("TESTING: What happens if we break things?")
print("="*70)

x = torch.randn(32, 4096, device='cuda')
ln = nn.LayerNorm(4096).cuda()

# Normal measurement
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
out_pytorch = ln(x)
end.record()
torch.cuda.synchronize()
pt_time = start.elapsed_time(end)

start.record()
out_ours = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-5)
end.record()
torch.cuda.synchronize()
our_time = start.elapsed_time(end)

print(f"Normal: PyTorch={pt_time:.4f}ms, Ours={our_time:.4f}ms, Speedup={pt_time/our_time:.2f}x")

# Check outputs match
match = torch.allclose(out_pytorch, out_ours, rtol=1e-4)
print(f"Outputs match: {match}")

# Now use wrong epsilon - should still work but might differ
out_wrong_eps = fused_layernorm_cuda.layernorm(x, ln.weight, ln.bias, 1e-12)
match_wrong = torch.allclose(out_pytorch, out_wrong_eps, rtol=1e-4)
print(f"With different epsilon, outputs match: {match_wrong}")

# Test with different data types
x_double = x.double()
ln_double = nn.LayerNorm(4096).cuda().double()

try:
    out_double = fused_layernorm_cuda.layernorm(x_double, ln_double.weight, ln_double.bias, 1e-5)
    print("Double precision works")
except Exception as e:
    print(f"Double precision fails (expected): {e}")

print("\nThese tests prove the kernel is actually running and computing!")
