# GPU Setup Guide for TubeScribe AI

This guide explains how to enable GPU acceleration for faster transcription and summarization.

## Benefits of GPU Acceleration

- **10-50x faster transcription** compared to CPU
- **5-10x faster summarization** and translation
- Real-time or faster processing for most videos
- Better resource utilization
- Lower CPU usage, allowing other tasks to run smoothly

## Quick Start

1. **Check if you have a compatible GPU** (see Prerequisites below)
2. **Install CUDA-enabled PyTorch** (see Installation Steps)
3. **Verify GPU setup** using `python check_gpu.py`
4. **Restart the backend** - GPU will be detected automatically

## Prerequisites

### For NVIDIA GPUs (CUDA):
1. **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
   - Most modern GPUs (2015+) are supported
   - Check your GPU: https://developer.nvidia.com/cuda-gpus
2. **NVIDIA drivers** (latest recommended)
   - Windows: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers)
   - Linux: Usually installed via package manager
3. **CUDA Toolkit** 11.8 or 12.1+ (recommended: 12.1)
   - Not strictly required if you only install PyTorch with CUDA support
   - Only needed if you compile CUDA code yourself

### Check if you have CUDA drivers:
```bash
nvidia-smi
```

**Windows (Command Prompt/PowerShell):**
```powershell
nvidia-smi
```

**Linux/Mac (with NVIDIA GPU):**
```bash
nvidia-smi
```

If this command works and shows your GPU information, you have CUDA drivers installed and can proceed with GPU setup.

## Installation Steps

### Step 1: Update requirements.txt

Edit `requirements.txt` and ensure GPU support is enabled:

**For CUDA 12.1 (Recommended - Latest):**
```txt
# Comment out CPU version:
# --extra-index-url https://download.pytorch.org/whl/cpu

# Uncomment GPU version:
--extra-index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (Older GPUs):**
```txt
# --extra-index-url https://download.pytorch.org/whl/cu121
--extra-index-url https://download.pytorch.org/whl/cu118
```

**Note:** The default `requirements.txt` already has CUDA 12.1 enabled. If you need CUDA 11.8, change `cu121` to `cu118`.

### Step 2: Install/Reinstall Dependencies

**If setting up for the first time:**
```bash
pip install -r requirements.txt
```

**If upgrading from CPU to GPU:**
```bash
pip install -r requirements.txt --upgrade --force-reinstall torch torchvision torchaudio
```

**Or install PyTorch manually first:**
```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

**Use the included script (Recommended):**
```bash
python check_gpu.py
```

**Windows (PowerShell):**
```powershell
python check_gpu.py
```

**Linux/Mac:**
```bash
python check_gpu.py
```

**Expected Output (GPU Detected):**
```
==================================================
GPU/CUDA Status Check
==================================================
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3060
GPU Memory: 12.0 GB
CUDA Version: 12.1
Number of GPUs: 1
[OK] GPU setup is correct!
==================================================
```

**Alternative: Quick Check**
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Automatic GPU Detection

The backend automatically detects and uses GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

When you start the backend, you'll see GPU detection in the logs:
```
🔧 Device: CUDA
🎮 GPU: NVIDIA GeForce RTX 3060
💾 GPU Memory: 12.0GB
✅ Transcription model loaded successfully on CUDA
✅ Summarization model loaded successfully on CUDA
```

**If GPU is not available, you'll see:**
```
🔧 Device: CPU
[INFO] Models will run on CPU (slower but functional)
✅ Transcription model loaded successfully on CPU
✅ Summarization model loaded successfully on CPU
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

**1. Verify NVIDIA drivers are installed:**
```bash
nvidia-smi
```
If this fails, install/update NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)

**2. Check PyTorch CUDA version:**
```bash
python -c "import torch; print('PyTorch CUDA Version:', torch.version.cuda); print('CUDA Available:', torch.cuda.is_available())"
```

**3. Verify CUDA toolkit (optional - only needed for development):**
```bash
nvcc --version
```
Note: CUDA toolkit is not required if you install PyTorch with CUDA support via pip.

**4. Reinstall PyTorch with correct CUDA version:**
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**5. Check for version conflicts:**
If you installed PyTorch from different sources, conflicts may occur. Use:
```bash
pip list | grep torch
```
Ensure all torch packages come from the same source.

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Use a smaller Whisper model** (in `app.py`):
   ```python
   # Change from:
   model="openai/whisper-base"  # ~500MB
   
   # To one of:
   model="openai/whisper-tiny"   # ~75MB, faster but less accurate
   model="openai/whisper-small"  # ~250MB, balance
   ```

2. **Close other GPU-intensive applications:**
   - Games, video editing software, other AI workloads
   - Check GPU usage: `nvidia-smi`

3. **Process shorter video segments:**
   - The backend automatically handles this, but very long videos may still cause issues

4. **Clear GPU cache** (restart backend):
   ```bash
   # Stop the backend (Ctrl+C)
   # Restart it
   python app.py
   # or
   uvicorn app:app --reload
   ```

5. **Reduce batch processing:**
   - The backend processes sequentially by default, but check for concurrent requests

### Fallback to CPU

If GPU fails to initialize, the backend **automatically falls back to CPU**. No action needed.

**Check logs for:**
```
🔧 Device: CPU
[INFO] CUDA not available, using CPU
```

**If you expected GPU:**
1. Run `python check_gpu.py` to diagnose
2. Check `nvidia-smi` works
3. Verify PyTorch CUDA installation (see above)

### Multiple GPUs

If you have multiple GPUs, PyTorch will use GPU 0 by default. To use a specific GPU, set environment variable:
```bash
# Windows
set CUDA_VISIBLE_DEVICES=1

# Linux/Mac
export CUDA_VISIBLE_DEVICES=1
```

Then restart the backend.

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

## Apple Silicon (M1/M2/M3/M4) Support

For Apple Silicon Macs, PyTorch can use Metal Performance Shaders (MPS) for GPU acceleration.

### Setup

1. **Install PyTorch with MPS support** (default installation includes MPS):
```bash
pip install torch torchvision torchaudio
```

2. **Verify MPS is available:**
```bash
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available()); print('MPS Built:', torch.backends.mps.is_built())"
```

3. **The backend will automatically detect and use MPS:**
```
🔧 Device: MPS
✅ Transcription model loaded successfully on MPS
✅ Summarization model loaded successfully on MPS
```

### Performance

- **2-5x faster** than CPU on Apple Silicon
- Similar speedup to CUDA (though slightly slower than high-end NVIDIA GPUs)
- Works automatically - no CUDA installation needed

### Troubleshooting MPS

**If MPS is not available:**
1. Update to macOS 12.3+ (required for MPS)
2. Reinstall PyTorch: `pip install --upgrade torch torchvision torchaudio`
3. Check PyTorch version: `python -c "import torch; print(torch.__version__)"` (needs 1.12+)

**MPS memory issues:**
- Similar to CUDA OOM errors
- Restart the backend to clear GPU memory
- Use smaller models if needed

## Monitoring GPU Usage

### NVIDIA GPUs

**Continuous monitoring:**
```bash
# Linux/Mac:
watch -n 1 nvidia-smi

# Or update every second:
nvidia-smi -l 1

# Windows (PowerShell):
# Run nvidia-smi in a separate window or use:
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

**One-time check:**
```bash
nvidia-smi
```

**What to look for:**
- **GPU-Util**: Should increase during transcription (0-100%)
- **Memory-Usage**: Shows VRAM usage (should increase with models loaded)
- **Processes**: Lists processes using the GPU

### Apple Silicon (MPS)

**Activity Monitor:**
1. Open Activity Monitor (Applications → Utilities)
2. Go to "Window" → "GPU History"
3. Monitor GPU usage while transcribing

**Terminal (if available):**
```bash
# System monitoring
top -pid $(pgrep -f "python.*app.py")
```

### Expected Behavior

**During transcription:**
- GPU utilization: 50-100% (depends on model size)
- GPU memory: Increases when models load, stabilizes during processing
- Temperature: May increase (normal)

**When idle:**
- GPU utilization: 0-5%
- GPU memory: Still allocated (models stay in memory for faster subsequent requests)

## Performance Tips

1. **First request is slower**: Models load into GPU memory on first use
2. **Keep backend running**: Models stay in GPU memory for faster subsequent requests
3. **Batch processing**: Process multiple videos sequentially for optimal GPU utilization
4. **Model selection**: Smaller models = faster processing, larger models = better accuracy

## Notes

- **Model loading**: First transcription after backend start loads models into GPU memory (~1-2GB)
- **Memory persistence**: GPU memory remains allocated until backend restarts (intentional - faster subsequent requests)
- **Concurrent requests**: Multiple requests share GPU resources and are processed sequentially
- **Production scaling**: For high throughput, consider:
  - Multiple GPU instances
  - Load balancing across multiple backends
  - GPU scheduling (Kubernetes GPU nodes)

## Which CUDA Version Should I Use?

| Your Setup | Recommended Version | Reason |
|------------|---------------------|--------|
| New GPU (2020+) | CUDA 12.1 | Latest, best performance |
| Older GPU (2015-2019) | CUDA 11.8 | Better compatibility |
| Unsure | CUDA 12.1 | Works on most modern GPUs |

**Quick check:** Run `nvidia-smi` and look at the top right - it shows the maximum CUDA version your drivers support.

## Common Issues Summary

| Issue | Solution |
|-------|----------|
| CUDA not available | Install/update NVIDIA drivers, reinstall PyTorch with CUDA |
| Out of memory | Use smaller model, close other GPU apps, restart backend |
| Slow performance | Verify GPU is actually being used (`nvidia-smi`), check GPU utilization |
| Wrong CUDA version | Match PyTorch CUDA version with your GPU drivers |

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
