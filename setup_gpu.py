"""Helper script to install PyTorch with CUDA support and verify GPU setup."""

import subprocess
import sys
import platform


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected via nvidia-smi")
            print(result.stdout.split('\n')[8:12])  # Show GPU info
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ nvidia-smi not found or timed out")
        print("  Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        return False


def check_pytorch_cuda():
    """Check if PyTorch has CUDA support."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        pytorch_version = torch.__version__
        
        print(f"\nPyTorch Version: {pytorch_version}")
        print(f"CUDA Available: {cuda_available}")
        
        if '+cpu' in pytorch_version:
            print("✗ CPU-only version detected (no GPU support)")
            return False
        elif cuda_available:
            print("✓ PyTorch with CUDA support is installed and working!")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ PyTorch with CUDA is installed but CUDA is not available")
            print("  This might indicate a driver or CUDA toolkit issue")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False


def install_pytorch_cuda():
    """Install PyTorch with CUDA 12.1 support (specific versions)."""
    print("\nInstalling PyTorch 2.5.1 with CUDA 12.1 support...")
    print("This may take a few minutes...\n")
    
    index_url = "https://download.pytorch.org/whl/cu121"
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'torch==2.5.1+cu121',
            'torchvision==0.20.1+cu121',
            'torchaudio==2.5.1+cu121',
            '--extra-index-url', index_url
        ], check=True)
        print("\n✓ PyTorch 2.5.1 with CUDA 12.1 installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nYou can manually install with:")
        print("  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \\")
        print("      --extra-index-url https://download.pytorch.org/whl/cu121")
        return False


def main():
    """Main function to check and setup GPU support."""
    print("=" * 70)
    print("GPU Setup and Verification")
    print("=" * 70)
    print()
    
    # Check for NVIDIA GPU
    has_gpu = check_nvidia_gpu()
    print()
    
    # Check PyTorch CUDA
    has_pytorch_cuda = check_pytorch_cuda()
    print()
    
    if has_pytorch_cuda:
        print("=" * 70)
        print("✓ GPU setup is complete! You can now use GPU for training.")
        print("=" * 70)
        return
    
    if not has_gpu:
        print("=" * 70)
        print("⚠ No NVIDIA GPU detected. You'll need to:")
        print("  1. Install/update NVIDIA drivers")
        print("  2. Ensure you have a CUDA-capable GPU")
        print("=" * 70)
        return
    
    # GPU detected but PyTorch CUDA not working
    print("=" * 70)
    print("GPU detected but PyTorch CUDA support is missing or not working.")
    print()
    
    response = input("Would you like to install PyTorch 2.5.1 with CUDA 12.1? (y/n): ").strip().lower()
    
    if response == 'y':
        # Uninstall existing PyTorch
        print("\nUninstalling existing PyTorch installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'], 
                      capture_output=True)
        
        # Install with CUDA 12.1
        if install_pytorch_cuda():
            print("\nVerifying installation...")
            check_pytorch_cuda()
    else:
        print("\nYou can manually install PyTorch with CUDA 12.1 using:")
        print("  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \\")
        print("      --extra-index-url https://download.pytorch.org/whl/cu121")


if __name__ == '__main__':
    main()

