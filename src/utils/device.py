"""Device utilities for GPU detection and management."""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        device: Preferred device ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        torch.device instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    return torch.device(device)


def check_gpu_available() -> bool:
    """Check if GPU is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_gpu_info() -> dict:
    """Get GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'available': torch.cuda.is_available(),
        'count': 0,
        'devices': []
    }
    
    if info['available']:
        info['count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        for i in range(info['count']):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                'allocated_memory_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'cached_memory_gb': torch.cuda.memory_reserved(i) / (1024**3),
                'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            }
            info['devices'].append(device_info)
    
    return info


def print_gpu_info():
    """Print GPU information to console."""
    info = get_gpu_info()
    
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    print(f"CUDA Available: {info['available']}")
    
    if info['available']:
        print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
        print(f"Number of GPUs: {info['count']}")
        
        for device in info['devices']:
            print(f"\nGPU {device['index']}: {device['name']}")
            print(f"  Total Memory: {device['total_memory_gb']:.2f} GB")
            print(f"  Allocated Memory: {device['allocated_memory_gb']:.2f} GB")
            print(f"  Cached Memory: {device['cached_memory_gb']:.2f} GB")
            print(f"  Compute Capability: {device['compute_capability']}")
    else:
        print("\nNo CUDA-capable GPU detected.")
        print("The model will run on CPU (slower performance).")
    
    print("=" * 60)

