"""
Visualization script for benchmark results.

Creates professional performance graphs and tables for the README.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from typing import Dict, List
import argparse

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
    def __init__(self, results_path: str, output_dir: str):
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        if results_path.endswith('.csv'):
            self.df = pd.read_csv(results_path)
        elif results_path.endswith('.json'):
            with open(results_path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        else:
            raise ValueError("Results file must be .csv or .json")
        
        # Set figure parameters for high quality
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_speedup_by_model_plot(self):
        """Create bar plot showing speedup by model."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Group by model and calculate mean speedup
        model_speedup = self.df.groupby('model_name').agg({
            'forward_speedup': 'mean',
            'backward_speedup': 'mean',
            'total_speedup': 'mean',
            'memory_reduction': 'mean'
        }).reset_index()
        
        # Sort by hidden size (extract from model name)
        model_order = ['BERT-Base', 'BERT-Large', 'GPT-2 Small', 'GPT-2 Medium', 
                      'GPT-2 Large', 'GPT-2 XL', 'GPT-3 Small', 'GPT-3 Medium',
                      'GPT-3 Large', 'GPT-3 XL', 'GPT-3 XXL', 'GPT-3 175B']
        model_speedup['model_name'] = pd.Categorical(
            model_speedup['model_name'], categories=model_order, ordered=True
        )
        model_speedup = model_speedup.sort_values('model_name')
        
        # Plot speedup
        x = np.arange(len(model_speedup))
        width = 0.25
        
        bars1 = ax1.bar(x - width, model_speedup['forward_speedup'], width, 
                        label='Forward', alpha=0.8)
        bars2 = ax1.bar(x, model_speedup['backward_speedup'], width, 
                        label='Backward', alpha=0.8)
        bars3 = ax1.bar(x + width, model_speedup['total_speedup'], width, 
                        label='Total', alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('LayerNorm Speedup by Model Architecture')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_speedup['model_name'], rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax1.axhline(y=1.4, color='green', linestyle='--', alpha=0.5, label='Target (1.4x)')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x', ha='center', va='bottom', fontsize=8)
        
        # Plot memory reduction
        bars4 = ax2.bar(x, model_speedup['memory_reduction'] * 100, alpha=0.8, 
                       color='coral')
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Memory Reduction (%)')
        ax2.set_title('Memory Reduction by Model Architecture')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_speedup['model_name'], rotation=45, ha='right')
        ax2.axhline(y=25, color='green', linestyle='--', alpha=0.5, label='Target (25%)')
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speedup_by_model.png'), 
                   bbox_inches='tight')
        plt.close()
    
    def create_speedup_vs_batch_size_plot(self):
        """Create line plot showing speedup vs batch size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Select representative models
        models = ['BERT-Base', 'GPT-2 Medium', 'GPT-3 Large']
        
        for model in models:
            model_df = self.df[self.df['model_name'] == model]
            
            # Group by batch size
            batch_speedup = model_df.groupby('batch_size').agg({
                'total_speedup': 'mean',
                'memory_reduction': 'mean'
            }).reset_index()
            
            ax1.plot(batch_speedup['batch_size'], batch_speedup['total_speedup'], 
                    marker='o', label=model, linewidth=2, markersize=8)
            ax2.plot(batch_speedup['batch_size'], batch_speedup['memory_reduction'] * 100, 
                    marker='o', label=model, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Speedup vs Batch Size')
        ax1.set_xscale('log', base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.4, color='green', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Reduction (%)')
        ax2.set_title('Memory Reduction vs Batch Size')
        ax2.set_xscale('log', base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=25, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speedup_vs_batch_size.png'), 
                   bbox_inches='tight')
        plt.close()
    
    def create_heatmap_speedup(self):
        """Create heatmap showing speedup across configurations."""
        # Filter for float32 to simplify
        df_fp32 = self.df[self.df['dtype'] == 'float32']
        
        # Pivot for heatmap
        pivot_speedup = df_fp32.pivot_table(
            values='total_speedup',
            index='hidden_size',
            columns='total_seq_len',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=1.0, vmin=0.8, vmax=1.8, cbar_kws={'label': 'Speedup Factor'})
        plt.title('Speedup Heatmap: Hidden Size vs Total Sequence Length')
        plt.xlabel('Total Sequence Length (Batch × Seq Length)')
        plt.ylabel('Hidden Size')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speedup_heatmap.png'), 
                   bbox_inches='tight')
        plt.close()
    
    def create_performance_table(self):
        """Create performance summary table for README."""
        # Select key configurations
        key_configs = [
            ('BERT-Base', 32, 512),
            ('BERT-Large', 32, 512),
            ('GPT-2 Medium', 16, 1024),
            ('GPT-2 Large', 16, 1024),
            ('GPT-3 Large', 8, 2048),
        ]
        
        table_data = []
        for model, batch_size, seq_len in key_configs:
            row_df = self.df[
                (self.df['model_name'] == model) & 
                (self.df['batch_size'] == batch_size) & 
                (self.df['seq_len'] == seq_len) &
                (self.df['dtype'] == 'float32')
            ]
            
            if not row_df.empty:
                row = row_df.iloc[0]
                table_data.append({
                    'Model': model,
                    'Batch': batch_size,
                    'Seq Len': seq_len,
                    'Hidden': row['hidden_size'],
                    'PyTorch (ms)': f"{row['pytorch_forward_ms'] + row['pytorch_backward_ms']:.2f}",
                    'Fused (ms)': f"{row['fused_forward_ms'] + row['fused_backward_ms']:.2f}",
                    'Speedup': f"{row['total_speedup']:.2f}x",
                    'Memory Reduction': f"{row['memory_reduction']*100:.1f}%"
                })
        
        # Create markdown table
        df_table = pd.DataFrame(table_data)
        markdown_table = df_table.to_markdown(index=False)
        
        # Save to file
        with open(os.path.join(self.output_dir, 'performance_table.md'), 'w') as f:
            f.write(markdown_table)
        
        return markdown_table
    
    def create_dtype_comparison(self):
        """Create comparison between FP32 and FP16 performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Group by dtype
        dtype_comparison = self.df.groupby(['dtype', 'model_name']).agg({
            'total_speedup': 'mean',
            'memory_reduction': 'mean'
        }).reset_index()
        
        # Select subset of models
        models = ['BERT-Base', 'GPT-2 Medium', 'GPT-3 Large']
        dtype_comparison = dtype_comparison[dtype_comparison['model_name'].isin(models)]
        
        # Pivot for plotting
        speedup_pivot = dtype_comparison.pivot(
            index='model_name', columns='dtype', values='total_speedup'
        )
        memory_pivot = dtype_comparison.pivot(
            index='model_name', columns='dtype', values='memory_reduction'
        )
        
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, speedup_pivot['float32'], width, 
                        label='FP32', alpha=0.8)
        bars2 = ax1.bar(x + width/2, speedup_pivot['float16'], width, 
                        label='FP16', alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Speedup Comparison: FP32 vs FP16')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x', ha='center', va='bottom')
        
        # Memory reduction comparison
        bars3 = ax2.bar(x - width/2, memory_pivot['float32'] * 100, width, 
                       label='FP32', alpha=0.8)
        bars4 = ax2.bar(x + width/2, memory_pivot['float16'] * 100, width, 
                       label='FP16', alpha=0.8)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Memory Reduction (%)')
        ax2.set_title('Memory Reduction: FP32 vs FP16')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dtype_comparison.png'), 
                   bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("Creating speedup by model plot...")
        self.create_speedup_by_model_plot()
        
        print("Creating speedup vs batch size plot...")
        self.create_speedup_vs_batch_size_plot()
        
        print("Creating speedup heatmap...")
        self.create_heatmap_speedup()
        
        print("Creating performance table...")
        table = self.create_performance_table()
        print("\nPerformance Table:")
        print(table)
        
        print("\nCreating dtype comparison...")
        self.create_dtype_comparison()
        
        print(f"\nAll visualizations saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results', type=str, 
                       default='benchmarks/results/benchmark_results.csv',
                       help='Path to benchmark results file')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmarks/results/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results, args.output_dir)
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()