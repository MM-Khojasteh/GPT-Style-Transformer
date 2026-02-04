"""GPU memory profiling utilities for monitoring training."""

import torch
from typing import Dict, Optional, List
from collections import defaultdict
import time


class MemoryProfiler:
    """Profiler for tracking GPU memory usage during training."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory profiler.
        
        Args:
            device: Device to profile (default: current CUDA device)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snapshots: List[Dict[str, float]] = []
        self.peak_memory = 0.0
        self.start_memory = 0.0
        
    def start(self):
        """Start profiling and record initial memory state."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_memory = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
            self.peak_memory = self.start_memory
        else:
            self.start_memory = 0.0
            self.peak_memory = 0.0
    
    def snapshot(self, label: str = "", additional_info: Optional[Dict] = None):
        """Take a memory snapshot.
        
        Args:
            label: Label for this snapshot
            additional_info: Additional information to record
        """
        if self.device.type != 'cuda':
            return
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)  # MB
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)  # MB
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**2)  # MB
        
        snapshot = {
            'label': label,
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'timestamp': time.time()
        }
        
        if additional_info:
            snapshot.update(additional_info)
        
        self.snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, peak_allocated)
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current memory statistics.
        
        Returns:
            Dictionary with current memory stats
        """
        if self.device.type != 'cuda':
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'peak_allocated_mb': 0.0,
                'peak_reserved_mb': 0.0
            }
        
        return {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved(self.device) / (1024**2),
            'peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / (1024**2),
            'peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / (1024**2)
        }
    
    def print_summary(self):
        """Print summary of memory usage."""
        if self.device.type != 'cuda':
            print("Memory profiling not available for CPU device")
            return
        
        print("=" * 70)
        print("Memory Profiling Summary")
        print("=" * 70)
        
        current = self.get_current_stats()
        print(f"\nCurrent Memory:")
        print(f"  Allocated: {current['allocated_mb']:.2f} MB")
        print(f"  Reserved: {current['reserved_mb']:.2f} MB")
        print(f"  Peak Allocated: {current['peak_allocated_mb']:.2f} MB")
        print(f"  Peak Reserved: {current['peak_reserved_mb']:.2f} MB")
        
        if self.snapshots:
            print(f"\nSnapshots ({len(self.snapshots)}):")
            for i, snap in enumerate(self.snapshots):
                label = snap.get('label', f'Snapshot {i+1}')
                print(f"  {label}:")
                print(f"    Allocated: {snap['allocated_mb']:.2f} MB")
                print(f"    Reserved: {snap['reserved_mb']:.2f} MB")
                if 'step' in snap:
                    print(f"    Step: {snap['step']}")
                if 'loss' in snap:
                    print(f"    Loss: {snap['loss']:.4f}")
        
        print()
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory
    
    def reset(self):
        """Reset profiler and clear snapshots."""
        self.snapshots = []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.start_memory = 0.0
        self.peak_memory = 0.0


def estimate_batch_size(
    model: torch.nn.Module,
    seq_len: int,
    embed_dim: int,
    available_memory_mb: float,
    target_utilization: float = 0.85
) -> int:
    """Estimate optimal batch size based on available memory.
    
    Args:
        model: Model to estimate for
        seq_len: Sequence length
        embed_dim: Embedding dimension
        available_memory_mb: Available GPU memory in MB
        target_utilization: Target memory utilization (0.0-1.0)
        
    Returns:
        Estimated optimal batch size
    """
    # Rough estimation: each sample uses approximately
    # seq_len * embed_dim * num_layers * bytes_per_element
    # For FP16: 2 bytes, for FP32: 4 bytes
    
    # Get model parameters count
    num_params = sum(p.numel() for p in model.parameters())
    
    # Model memory (FP16)
    model_memory_mb = (num_params * 2) / (1024**2)
    
    # Optimizer memory (FP32, AdamW: 2 states)
    optimizer_memory_mb = (num_params * 4 * 2) / (1024**2)
    
    # Available for activations
    target_memory_mb = available_memory_mb * target_utilization
    activation_memory_mb = target_memory_mb - model_memory_mb - optimizer_memory_mb
    
    if activation_memory_mb <= 0:
        return 1  # Minimum batch size
    
    # Rough estimate: activation memory per sample
    # This is a simplified estimate - actual usage depends on many factors
    bytes_per_token = embed_dim * 2  # FP16
    memory_per_sample_mb = (seq_len * bytes_per_token * 4) / (1024**2)  # Rough estimate
    
    estimated_batch_size = int(activation_memory_mb / memory_per_sample_mb)
    
    # Clamp to reasonable range
    return max(1, min(estimated_batch_size, 128))


def print_memory_info(device: Optional[torch.device] = None):
    """Print current GPU memory information.
    
    Args:
        device: Device to check (default: current CUDA device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("Memory info not available for CPU device")
        return
    
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    
    print(f"GPU Memory ({device}):")
    print(f"  Allocated: {allocated:.2f} MB ({allocated/total*100:.2f}%)")
    print(f"  Reserved: {reserved:.2f} MB ({reserved/total*100:.2f}%)")
    print(f"  Total: {total:.2f} MB")
    print(f"  Free: {total - reserved:.2f} MB ({(total-reserved)/total*100:.2f}%)")

