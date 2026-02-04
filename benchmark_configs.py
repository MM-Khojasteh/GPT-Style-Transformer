"""Benchmark different model configurations and compare performance."""

import torch
import time
import argparse
from typing import Dict, List
from src.models.gpt import AdvancedGPT
from src.utils.config_presets import get_model_config, get_training_config, estimate_model_size
from src.utils.device import get_device, print_gpu_info, get_gpu_info
from src.utils.memory_profiler import MemoryProfiler, print_memory_info
from src.utils.tokenizer import CharTokenizer
from src.optimizations.gradient_checkpoint import apply_gradient_checkpointing


def benchmark_model(
    preset: str,
    vocab_size: int = 36,
    seq_len: int = 64,
    batch_size: int = 4,
    num_iterations: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """Benchmark a model configuration.
    
    Args:
        preset: Model preset name
        vocab_size: Vocabulary size
        seq_len: Sequence length for benchmarking
        batch_size: Batch size for benchmarking
        num_iterations: Number of forward passes to average
        device: Device to run on
        
    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = get_device()
    
    print(f"\n{'='*70}")
    print(f"Benchmarking {preset.upper()} Configuration")
    print(f"{'='*70}")
    
    # Get model configuration
    model_config = get_model_config(preset=preset, vocab_size=vocab_size)
    
    # Create model
    model = AdvancedGPT(**model_config)
    model = model.to(device)
    model.eval()
    
    # Get size estimates
    size_info = estimate_model_size(model_config)
    actual_params = model.get_num_params()
    
    print(f"\nModel Configuration:")
    print(f"  Parameters: {actual_params:,}")
    print(f"  Estimated Memory: {size_info['total_memory_mb']:.2f} MB")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  num_layers: {model_config['num_layers']}")
    if model_config.get('use_gqa'):
        print(f"  GQA: {model_config.get('num_kv_heads')} KV heads")
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Memory profiling
    profiler = MemoryProfiler(device=device)
    profiler.start()
    
    # Benchmark forward pass
    print(f"\nBenchmarking forward pass ({num_iterations} iterations)...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iterations):
            _ = model(input_ids)
            if device.type == 'cuda' and i == 0:
                profiler.snapshot(label="After first forward")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    avg_time = elapsed / num_iterations
    
    # Get memory stats
    profiler.snapshot(label="After benchmark")
    memory_stats = profiler.get_current_stats()
    
    # Compute throughput
    tokens_per_second = (batch_size * seq_len * num_iterations) / elapsed
    
    results = {
        'preset': preset,
        'parameters': actual_params,
        'estimated_memory_mb': size_info['total_memory_mb'],
        'actual_memory_mb': memory_stats.get('peak_allocated_mb', 0),
        'forward_time_ms': avg_time * 1000,
        'throughput_tokens_per_sec': tokens_per_second,
        'throughput_samples_per_sec': batch_size * num_iterations / elapsed,
        'memory_efficiency': (size_info['total_memory_mb'] / memory_stats.get('peak_allocated_mb', 1)) if memory_stats.get('peak_allocated_mb', 0) > 0 else 0
    }
    
    print(f"\nResults:")
    print(f"  Average forward time: {results['forward_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
    if device.type == 'cuda':
        print(f"  Peak memory: {results['actual_memory_mb']:.2f} MB")
    
    profiler.print_summary()
    
    return results


def compare_configs(
    presets: List[str] = ['small', 'medium', 'large'],
    vocab_size: int = 36,
    seq_len: int = 64,
    batch_size: int = 4,
    num_iterations: int = 10
):
    """Compare multiple model configurations.
    
    Args:
        presets: List of preset names to compare
        vocab_size: Vocabulary size
        seq_len: Sequence length
        batch_size: Batch size
        num_iterations: Number of iterations per benchmark
    """
    device = get_device()
    
    print("=" * 70)
    print("Model Configuration Benchmark")
    print("=" * 70)
    
    print_gpu_info()
    print()
    
    results = []
    
    for preset in presets:
        try:
            result = benchmark_model(
                preset=preset,
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=batch_size,
                num_iterations=num_iterations,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {preset}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    
    if not results:
        print("No results to compare")
        return
    
    # Header
    print(f"\n{'Preset':<10} {'Params':<15} {'Memory (MB)':<15} {'Time (ms)':<12} {'Tokens/s':<12}")
    print("-" * 70)
    
    # Results
    for r in results:
        print(f"{r['preset']:<10} {r['parameters']:>14,} {r['actual_memory_mb']:>14.2f} "
              f"{r['forward_time_ms']:>11.2f} {r['throughput_tokens_per_sec']:>11.0f}")
    
    # Find best
    fastest = min(results, key=lambda x: x['forward_time_ms'])
    most_efficient = min(results, key=lambda x: x['actual_memory_mb'] / x['parameters'] if x['parameters'] > 0 else float('inf'))
    
    print("\nBest Performance:")
    print(f"  Fastest: {fastest['preset']} ({fastest['forward_time_ms']:.2f} ms)")
    print(f"  Most Memory Efficient: {most_efficient['preset']} "
          f"({most_efficient['actual_memory_mb']/most_efficient['parameters']*1e6:.4f} MB per 1M params)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark model configurations')
    parser.add_argument('--presets', nargs='+', default=['small', 'medium', 'large'],
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Presets to benchmark')
    parser.add_argument('--vocab-size', type=int, default=36,
                        help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per benchmark')
    args = parser.parse_args()
    
    compare_configs(
        presets=args.presets,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )


if __name__ == '__main__':
    main()

