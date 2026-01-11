# Quick Start Guide - TubeScribe AI

Follow these steps to get TubeScribe AI running on your machine.

## Prerequisites Checklist

- [ ] **Python 3.10+** installed (required)
- [ ] **Node.js 18+** installed (for frontend only)
- [ ] **FFmpeg** installed (required for audio processing)
- [ ] **Git** (if cloning from repository)

## Step-by-Step Setup

### Step 1: Install Frontend Dependencies

Open a terminal in the project root and run:

```bash
npm install
```

### Step 2: Set Up Python Backend

**Option A: Automatic Setup (Recommended for Windows)**

```powershell
# PowerShell
.\scripts\fix-permissions.ps1
```

Or:

```cmd
# Command Prompt
scripts\fix-permissions.bat
```

**Option B: Manual Setup**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

**Frontend** - Create `.env.local` in the root directory:
```env
VITE_API_KEY=your_gemini_api_key_here
```

> **Note**: You don't need Gemini API keys for transcription and summarization! Only needed for translation and chat features.

### Step 4: Start All Services

You need to run **2 services** in separate terminal windows:

#### Terminal 1: Python Backend (Port 8000)

**Windows:**
```powershell
.\scripts\start-python-backend.ps1
```

Or:
```cmd
scripts\start-python-backend.bat
```

**Linux/Mac:**
```bash
./scripts/start-python-backend.sh
```

**Manual Start:**
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

✅ You should see: `Uvicorn running on http://0.0.0.0:8000`

#### Terminal 2: Frontend (Port 5173)

```bash
npm run dev
```

✅ You should see: `Local: http://localhost:5173/`

## Step 5: Verify Everything Works

1. **Python Backend**: Open http://localhost:8000/health
   - Should return: `{"status": "healthy", "models_ready": {...}}`

2. **Frontend**: Open http://localhost:5173
   - Should show the TubeScribe AI interface

## Quick Test

1. Open http://localhost:5173 in your browser
2. Paste a YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
3. Click "Generate Transcript"
4. Wait for transcription (Python backend will process the video)
5. Click "Summarize" to get summary with keywords

## Troubleshooting

### Port Already in Use

If a port is already in use:

**Python Backend (8000):**
- Change port in script or command: `--port 8001`


**Frontend (5173):**
- Vite will automatically use the next available port

### Python Backend Not Starting

1. Check if virtual environment is activated
2. Verify dependencies: `pip list | findstr fastapi`
3. Check for errors in terminal output

### Models Not Loading

First run will download models (~8GB total):
- Whisper: ~500MB
- mBART: ~2GB
- KeyBERT: ~500MB
- Plus dependencies

Be patient on first startup!

## Need Help?

- Check [PYTHON_BACKEND_README.md](PYTHON_BACKEND_README.md) for Python backend details
- Check main [README.md](README.md) for full documentation
- Review terminal error messages for specific issues

## Summary

```
┌─────────────────────────────────────────────┐
│  TubeScribe AI - Service Overview          │
├─────────────────────────────────────────────┤
│                                             │
│  Terminal 1: Python Backend   → :8000      │
│  Terminal 2: Frontend (React) → :5173      │
│                                             │
│  Open browser: http://localhost:5173       │
│                                             │
└─────────────────────────────────────────────┘
```

That's it! You're ready to transcribe, summarize, and analyze YouTube videos! 🚀
