"""Quick script to check GPU/CUDA availability for TubeScribe AI"""
import torch

print("=" * 50)
print("GPU/CUDA Status Check")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("[OK] GPU setup is correct!")
else:
    print("[WARNING] CUDA not available - will use CPU")
    print("\nIf you expected GPU support:")
    print("1. Check if NVIDIA GPU is installed")
    print("2. Verify CUDA drivers: nvidia-smi")
    print("3. Reinstall PyTorch with CUDA support")

print("=" * 50)
