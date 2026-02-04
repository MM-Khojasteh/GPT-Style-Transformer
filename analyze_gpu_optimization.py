"""Analyze GPU utilization and suggest optimal architecture improvements."""

import torch
from src.models.gpt import AdvancedGPT
from src.utils.device import get_gpu_info


def estimate_memory_usage(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, batch_size, seq_len):
    """Estimate GPU memory usage for a model configuration."""
    # Rough estimation formula
    # Model parameters (in bytes, FP16)
    param_memory = (
        vocab_size * embed_dim * 2 +  # Token embeddings
        max_seq_len * embed_dim * 2 +  # Positional embeddings
        num_layers * (
            4 * embed_dim * embed_dim * 2 +  # Attention Q, K, V, O projections
            embed_dim * embed_dim * 2 +  # FFN up projection
            embed_dim * embed_dim * 2 +  # FFN gate projection
            embed_dim * embed_dim * 2 +  # FFN down projection
            2 * embed_dim * 2  # Layer norms
        ) +
        embed_dim * embed_dim * 2 +  # Final layer norm
        vocab_size * embed_dim * 2  # Output head (if not weight tied)
    ) / (1024**2)  # Convert to MB
    
    # Activation memory (per batch, FP16)
    activation_memory = (
        batch_size * seq_len * embed_dim * 2 * num_layers * 4  # Rough estimate
    ) / (1024**2)
    
    # Optimizer states (AdamW: 2x model params in FP32)
    optimizer_memory = param_memory * 2 * 2  # 2x for Adam states, 2x for FP32
    
    total = param_memory + activation_memory + optimizer_memory
    return {
        'param_memory_mb': param_memory,
        'activation_memory_mb': activation_memory,
        'optimizer_memory_mb': optimizer_memory,
        'total_memory_mb': total,
        'total_memory_gb': total / 1024
    }


def find_optimal_config(gpu_memory_gb=6.0, vocab_size=36, target_utilization=0.85):
    """Find optimal model configuration for given GPU memory."""
    available_memory_mb = gpu_memory_gb * 1024 * target_utilization
    
    print("=" * 70)
    print("GPU Memory Analysis & Architecture Optimization")
    print("=" * 70)
    print(f"\nGPU Memory: {gpu_memory_gb} GB")
    print(f"Target Utilization: {target_utilization * 100:.0f}%")
    print(f"Available Memory: {available_memory_mb:.0f} MB")
    print()
    
    # Current configuration
    print("Current Configuration:")
    print("-" * 70)
    current = estimate_memory_usage(
        vocab_size=vocab_size,
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        batch_size=4,
        seq_len=64
    )
    print(f"  embed_dim: 128")
    print(f"  num_heads: 4")
    print(f"  num_layers: 4")
    print(f"  max_seq_len: 128")
    print(f"  batch_size: 4")
    print(f"  seq_len: 64")
    print(f"  Parameters: ~1.1M")
    print(f"  Memory Usage: {current['total_memory_mb']:.2f} MB ({current['total_memory_gb']:.3f} GB)")
    print(f"  GPU Utilization: {(current['total_memory_mb'] / (gpu_memory_gb * 1024)) * 100:.2f}%")
    print()
    
    # Suggested configurations
    configs = [
        {
            'name': 'Small (Current)',
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'max_seq_len': 128,
            'batch_size': 4,
            'seq_len': 64
        },
        {
            'name': 'Medium - Balanced',
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'max_seq_len': 512,
            'batch_size': 8,
            'seq_len': 256
        },
        {
            'name': 'Large - Memory Optimized',
            'embed_dim': 384,
            'num_heads': 12,
            'num_layers': 8,
            'max_seq_len': 512,
            'batch_size': 6,
            'seq_len': 256
        },
        {
            'name': 'XL - Maximum Performance',
            'embed_dim': 512,
            'num_heads': 16,
            'num_layers': 10,
            'max_seq_len': 1024,
            'batch_size': 4,
            'seq_len': 512
        },
        {
            'name': 'XXL - With Gradient Checkpointing',
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12,
            'max_seq_len': 1024,
            'batch_size': 2,
            'seq_len': 512
        }
    ]
    
    print("Suggested Configurations:")
    print("=" * 70)
    
    for config in configs:
        mem = estimate_memory_usage(
            vocab_size=vocab_size,
            max_seq_len=config['max_seq_len'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            batch_size=config['batch_size'],
            seq_len=config['seq_len']
        )
        
        # Estimate parameters
        model = AdvancedGPT(
            vocab_size=vocab_size,
            max_seq_len=config['max_seq_len'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            weight_tying=True
        )
        num_params = model.get_num_params()
        
        utilization = (mem['total_memory_mb'] / (gpu_memory_gb * 1024)) * 100
        feasible = mem['total_memory_mb'] <= available_memory_mb
        
        print(f"\n{config['name']}:")
        print(f"  Configuration:")
        print(f"    embed_dim: {config['embed_dim']}")
        print(f"    num_heads: {config['num_heads']}")
        print(f"    num_layers: {config['num_layers']}")
        print(f"    max_seq_len: {config['max_seq_len']}")
        print(f"    batch_size: {config['batch_size']}")
        print(f"    seq_len: {config['seq_len']}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Memory Breakdown:")
        print(f"    Model params: {mem['param_memory_mb']:.2f} MB")
        print(f"    Activations: {mem['activation_memory_mb']:.2f} MB")
        print(f"    Optimizer: {mem['optimizer_memory_mb']:.2f} MB")
        print(f"    Total: {mem['total_memory_mb']:.2f} MB ({mem['total_memory_gb']:.3f} GB)")
        print(f"  GPU Utilization: {utilization:.2f}%")
        print(f"  Feasible: {'[YES]' if feasible else '[NO - exceeds available memory]'}")
    
    print()
    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("""
1. **Immediate Improvements (Easy Wins):**
   - Increase batch_size from 4 to 8-16 (better GPU utilization)
   - Increase seq_len from 64 to 256-512 (better context understanding)
   - Increase embed_dim from 128 to 256-384 (better representation capacity)

2. **Medium-term Improvements:**
   - Scale to Medium or Large configuration
   - Use gradient checkpointing for larger models
   - Increase max_seq_len to 512-1024 for longer context

3. **Advanced Optimizations:**
   - Use Grouped Query Attention (GQA) to reduce memory
   - Enable gradient checkpointing for XXL models
   - Use Flash Attention if available
   - Consider Mixture of Experts (MoE) for even larger models

4. **Training Optimizations:**
   - Increase gradient_accumulation_steps for effective larger batch sizes
   - Use mixed precision (already enabled)
   - Optimize data loading with multiple workers
   - Use learning rate warmup and cosine annealing
    """)


if __name__ == '__main__':
    gpu_info = get_gpu_info()
    gpu_memory_gb = gpu_info['devices'][0]['total_memory_gb'] if gpu_info['available'] else 6.0
    
    find_optimal_config(gpu_memory_gb=gpu_memory_gb, vocab_size=36)

