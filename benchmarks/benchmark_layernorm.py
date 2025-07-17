"""
Comprehensive benchmarking script for Fused LayerNorm CUDA operator.

This script benchmarks our implementation against PyTorch's native LayerNorm
across various model configurations (BERT, GPT-2, GPT-3) and batch sizes.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our fused implementation
from fused_layernorm import FusedLayerNorm


class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Model configurations (hidden_size, model_name)
    MODEL_CONFIGS = [
        (768, "BERT-Base"),
        (1024, "BERT-Large"),
        (768, "GPT-2 Small"),
        (1024, "GPT-2 Medium"),
        (1280, "GPT-2 Large"),
        (1600, "GPT-2 XL"),
        (2048, "GPT-3 Small"),
        (2560, "GPT-3 Medium"),
        (4096, "GPT-3 Large"),
        (5120, "GPT-3 XL"),
        (8192, "GPT-3 XXL"),
        (12288, "GPT-3 175B"),
    ]
    
    # Batch sizes to test
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Sequence lengths for transformer models
    SEQUENCE_LENGTHS = [128, 256, 512, 1024]
    
    # Number of warmup iterations
    WARMUP_ITERS = 50
    
    # Number of benchmark iterations
    BENCHMARK_ITERS = 200
    
    # Data types to test
    DTYPES = [torch.float32, torch.float16]


class LayerNormBenchmark:
    """Benchmark harness for LayerNorm implementations."""
    
    def __init__(self, device='cuda', verbose=True):
        self.device = device
        self.verbose = verbose
        self.results = []
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Get GPU info
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        if self.verbose:
            print(f"GPU: {self.gpu_name}")
            print(f"Memory: {self.gpu_memory:.1f} GB")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA: {torch.version.cuda}")
            print("-" * 60)
    
    def benchmark_forward(self, layer: nn.Module, input_tensor: torch.Tensor, 
                         num_iters: int) -> float:
        """Benchmark forward pass."""
        # Warmup
        for _ in range(BenchmarkConfig.WARMUP_ITERS):
            _ = layer(input_tensor)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iters):
            _ = layer(input_tensor)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / num_iters
        
        return elapsed_time
    
    def benchmark_backward(self, layer: nn.Module, input_tensor: torch.Tensor, 
                          num_iters: int) -> float:
        """Benchmark backward pass."""
        input_tensor.requires_grad = True
        
        # Warmup
        for _ in range(BenchmarkConfig.WARMUP_ITERS):
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iters):
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / num_iters
        
        return elapsed_time
    
    def measure_memory(self, layer: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Measure memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure forward pass memory
        start_memory = torch.cuda.memory_allocated()
        output = layer(input_tensor)
        torch.cuda.synchronize()
        forward_memory = torch.cuda.memory_allocated() - start_memory
        
        # Measure backward pass memory
        if input_tensor.requires_grad:
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()
            total_memory = torch.cuda.memory_allocated() - start_memory
            backward_memory = total_memory - forward_memory
        else:
            backward_memory = 0
            total_memory = forward_memory
        
        peak_memory = torch.cuda.max_memory_allocated() - start_memory
        
        return {
            'forward_memory_mb': forward_memory / 1e6,
            'backward_memory_mb': backward_memory / 1e6,
            'total_memory_mb': total_memory / 1e6,
            'peak_memory_mb': peak_memory / 1e6
        }
    
    def run_single_benchmark(self, batch_size: int, seq_len: int, hidden_size: int,
                           dtype: torch.dtype, model_name: str) -> Dict:
        """Run a single benchmark configuration."""
        # Create input tensor
        shape = (batch_size * seq_len, hidden_size)
        input_tensor = torch.randn(shape, device=self.device, dtype=dtype)
        
        # Create layers
        pytorch_layer = nn.LayerNorm(hidden_size, dtype=dtype).to(self.device)
        fused_layer = FusedLayerNorm(hidden_size, dtype=dtype).to(self.device)
        
        # Copy weights to ensure fair comparison
        with torch.no_grad():
            fused_layer.weight.data.copy_(pytorch_layer.weight.data)
            fused_layer.bias.data.copy_(pytorch_layer.bias.data)
        
        # Benchmark forward pass
        pytorch_forward_time = self.benchmark_forward(
            pytorch_layer, input_tensor, BenchmarkConfig.BENCHMARK_ITERS
        )
        fused_forward_time = self.benchmark_forward(
            fused_layer, input_tensor, BenchmarkConfig.BENCHMARK_ITERS
        )
        
        # Benchmark backward pass
        pytorch_backward_time = self.benchmark_backward(
            pytorch_layer, input_tensor.clone(), BenchmarkConfig.BENCHMARK_ITERS
        )
        fused_backward_time = self.benchmark_backward(
            fused_layer, input_tensor.clone(), BenchmarkConfig.BENCHMARK_ITERS
        )
        
        # Measure memory
        pytorch_memory = self.measure_memory(pytorch_layer, input_tensor.clone())
        fused_memory = self.measure_memory(fused_layer, input_tensor.clone())
        
        # Calculate speedups
        forward_speedup = pytorch_forward_time / fused_forward_time
        backward_speedup = pytorch_backward_time / fused_backward_time
        total_speedup = (pytorch_forward_time + pytorch_backward_time) / \
                       (fused_forward_time + fused_backward_time)
        
        # Calculate memory reduction
        memory_reduction = 1 - (fused_memory['peak_memory_mb'] / pytorch_memory['peak_memory_mb'])
        
        # Verify correctness
        with torch.no_grad():
            pytorch_output = pytorch_layer(input_tensor)
            fused_output = fused_layer(input_tensor)
            max_diff = torch.max(torch.abs(pytorch_output - fused_output)).item()
        
        result = {
            'model_name': model_name,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'total_seq_len': batch_size * seq_len,
            'dtype': str(dtype).split('.')[-1],
            'pytorch_forward_ms': pytorch_forward_time,
            'fused_forward_ms': fused_forward_time,
            'forward_speedup': forward_speedup,
            'pytorch_backward_ms': pytorch_backward_time,
            'fused_backward_ms': fused_backward_time,
            'backward_speedup': backward_speedup,
            'total_speedup': total_speedup,
            'pytorch_memory_mb': pytorch_memory['peak_memory_mb'],
            'fused_memory_mb': fused_memory['peak_memory_mb'],
            'memory_reduction': memory_reduction,
            'max_diff': max_diff,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_benchmarks(self, output_dir: str = 'benchmarks/results'):
        """Run comprehensive benchmarks."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress tracking
        total_configs = (len(BenchmarkConfig.MODEL_CONFIGS) * 
                        len(BenchmarkConfig.BATCH_SIZES) * 
                        len(BenchmarkConfig.SEQUENCE_LENGTHS) * 
                        len(BenchmarkConfig.DTYPES))
        
        pbar = tqdm(total=total_configs, desc="Running benchmarks")
        
        for hidden_size, model_name in BenchmarkConfig.MODEL_CONFIGS:
            for batch_size in BenchmarkConfig.BATCH_SIZES:
                for seq_len in BenchmarkConfig.SEQUENCE_LENGTHS:
                    for dtype in BenchmarkConfig.DTYPES:
                        # Skip configurations that would exceed memory
                        total_elements = batch_size * seq_len * hidden_size
                        bytes_needed = total_elements * (2 if dtype == torch.float16 else 4)
                        if bytes_needed > 8e9:  # Skip if > 8GB
                            pbar.update(1)
                            continue
                        
                        try:
                            result = self.run_single_benchmark(
                                batch_size, seq_len, hidden_size, dtype, model_name
                            )
                            self.results.append(result)
                            
                            # Print summary for significant results
                            if self.verbose and result['total_speedup'] > 1.3:
                                print(f"\n{model_name} (BS={batch_size}, Seq={seq_len}, {result['dtype']}): "
                                      f"{result['total_speedup']:.2f}x speedup, "
                                      f"{result['memory_reduction']*100:.1f}% memory reduction")
                        
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                if self.verbose:
                                    print(f"\nSkipping {model_name} BS={batch_size} Seq={seq_len} - OOM")
                            else:
                                raise e
                        
                        pbar.update(1)
                        torch.cuda.empty_cache()
        
        pbar.close()
        
        # Save results
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'benchmark_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary statistics
        summary = self.generate_summary(df)
        summary_path = os.path.join(output_dir, 'benchmark_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from benchmark results."""
        summary = {
            'gpu': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'num_benchmarks': len(df),
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'mean_forward_speedup': df['forward_speedup'].mean(),
                'mean_backward_speedup': df['backward_speedup'].mean(),
                'mean_total_speedup': df['total_speedup'].mean(),
                'mean_memory_reduction': df['memory_reduction'].mean(),
                'max_speedup': df['total_speedup'].max(),
                'min_speedup': df['total_speedup'].min(),
            },
            'by_model': {},
            'by_dtype': {}
        }
        
        # Summary by model
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            summary['by_model'][model] = {
                'mean_speedup': model_df['total_speedup'].mean(),
                'mean_memory_reduction': model_df['memory_reduction'].mean(),
                'num_configs': len(model_df)
            }
        
        # Summary by dtype
        for dtype in df['dtype'].unique():
            dtype_df = df[df['dtype'] == dtype]
            summary['by_dtype'][dtype] = {
                'mean_speedup': dtype_df['total_speedup'].mean(),
                'mean_memory_reduction': dtype_df['memory_reduction'].mean(),
                'num_configs': len(dtype_df)
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark Fused LayerNorm')
    parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print verbose output')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer configurations')
    
    args = parser.parse_args()
    
    # Adjust configurations for quick mode
    if args.quick:
        BenchmarkConfig.MODEL_CONFIGS = BenchmarkConfig.MODEL_CONFIGS[:4]
        BenchmarkConfig.BATCH_SIZES = [8, 32, 64]
        BenchmarkConfig.SEQUENCE_LENGTHS = [256, 512]
        BenchmarkConfig.BENCHMARK_ITERS = 50
    
    # Run benchmarks
    benchmark = LayerNormBenchmark(verbose=args.verbose)
    results_df = benchmark.run_benchmarks(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Average speedup: {results_df['total_speedup'].mean():.2f}x")
    print(f"Average memory reduction: {results_df['memory_reduction'].mean()*100:.1f}%")
    print(f"Best speedup: {results_df['total_speedup'].max():.2f}x")
    print(f"Worst speedup: {results_df['total_speedup'].min():.2f}x")
    print("="*60)


if __name__ == '__main__':
    main()