"""Check GPU availability and display information."""

import torch
import sys


def check_gpu_availability():
    """Check and display GPU availability information."""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test GPU computation
        print("\n" + "=" * 60)
        print("Testing GPU Computation")
        print("=" * 60)
        try:
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print("[OK] GPU computation test passed!")
            print(f"  Result shape: {z.shape}")
            print(f"  Result device: {z.device}")
        except Exception as e:
            print(f"[ERROR] GPU computation test failed: {e}")
    else:
        print("\nNo CUDA-capable GPU detected.")
        print("The model will run on CPU (slower performance).")
        print("\nTo use GPU:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA toolkit")
        print("  3. Install PyTorch with CUDA support")
    
    print("\n" + "=" * 60)
    print("PyTorch Information")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    return cuda_available


if __name__ == "__main__":
    check_gpu_availability()

