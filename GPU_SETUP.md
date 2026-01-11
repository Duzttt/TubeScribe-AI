# GPU Setup Guide for TubeScribe AI

This guide explains how to enable GPU acceleration for faster transcription.

## Benefits of GPU Acceleration

- **10-50x faster transcription** compared to CPU
- Real-time or faster processing for most videos
- Better resource utilization

## Prerequisites

### For NVIDIA GPUs (CUDA):
1. **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
2. **CUDA Toolkit** 11.8 or 12.1
3. **cuDNN** library (usually comes with CUDA)
4. **NVIDIA drivers** (latest recommended)

### Check if you have CUDA:
```bash
nvidia-smi
```

If this command works and shows your GPU, you have CUDA installed.

## Installation Steps

### Step 1: Install CUDA-enabled PyTorch

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Update requirements.txt

Edit `requirements.txt` and change:
```txt
# Comment out CPU version:
# --extra-index-url https://download.pytorch.org/whl/cpu

# Uncomment GPU version:
--extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Reinstall Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Step 4: Verify GPU Setup

**PowerShell:**
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Or use a Python script:**
Create a file `check_gpu.py`:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("GPU: N/A (Using CPU)")
```

Then run:
```powershell
python check_gpu.py
```

You should see:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
```

## Automatic GPU Detection

The backend automatically detects and uses GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

When you start the backend, you'll see:
```
🔧 Device: CUDA
🎮 GPU: NVIDIA GeForce RTX 3060
💾 GPU Memory: 12.0GB
✅ Transcription model loaded successfully on CUDA
```

## Performance Comparison

| Video Length | CPU Time | GPU Time (RTX 3060) | Speedup |
|--------------|----------|---------------------|---------|
| 1 minute     | ~2 min   | ~5 sec              | 24x     |
| 5 minutes    | ~10 min  | ~25 sec             | 24x     |
| 10 minutes   | ~20 min  | ~50 sec             | 24x     |
| 1 hour       | ~120 min | ~5 min              | 24x     |

*Results may vary based on GPU model*

## Troubleshooting

### "CUDA not available" but GPU exists
1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Ensure PyTorch CUDA version matches installed CUDA version

### Out of Memory Errors
If you get CUDA out of memory errors:
1. Use a smaller Whisper model (change `whisper-base` to `whisper-tiny` in `app.py`)
2. Reduce batch size
3. Close other GPU-intensive applications

### Fallback to CPU
If GPU fails to initialize, the backend automatically falls back to CPU. Check logs:
```
WARNING: CUDA not available, falling back to CPU
```

## Alternative: Using Smaller Models

For systems with limited GPU memory, use smaller Whisper models:

In `app.py`, change:
```python
model="openai/whisper-base"  # ~500MB
```

To:
```python
model="openai/whisper-tiny"  # ~75MB, faster but less accurate
```

Or:
```python
model="openai/whisper-small"  # ~250MB, balance
```

## Apple Silicon (M1/M2/M3) Support

For Apple Silicon Macs, PyTorch can use Metal Performance Shaders (MPS):

1. Install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

2. The backend will automatically detect and use MPS if available:
```
🔧 Device: MPS
✅ Transcription model loaded successfully on MPS
```

MPS provides similar speedup to CUDA on Apple Silicon.

## Monitoring GPU Usage

While transcribing, monitor GPU usage:
```bash
# On Linux/Mac:
watch -n 1 nvidia-smi

# Or continuously:
nvidia-smi -l 1
```

You should see GPU utilization during transcription.

## Notes

- First transcription after starting the backend may be slower (model loading)
- GPU memory persists until backend restarts
- Multiple concurrent requests will share GPU resources
- For production, consider using multiple GPU instances for better throughput
