# TubeScribe AI - Quick Setup Guide

This guide helps you get all three services running for the complete TubeScribe AI experience.

## 🚀 Quick Start (All Services)

### Step 1: Frontend Setup
```bash
npm install
# Create .env.local with: VITE_API_KEY=your_key (optional - only for translation/chat)
npm run dev
```
Frontend runs on: **http://localhost:5173**

### Step 2: Python Backend (Required - Transcription, Summarization, Keywords)

**Windows (PowerShell - Recommended):**
```powershell
.\scripts\start-python-backend.ps1
```

**Windows (Command Prompt):**
```bash
scripts\start-python-backend.bat
```

**Linux/Mac:**
```bash
./scripts/start-python-backend.sh
```

Python backend runs on: **http://localhost:8000**

## 📋 Service Overview

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| Frontend | 5173 | User Interface | ✅ Yes |
| Python Backend | 8000 | Transcription + Summarization + Keywords | ✅ Yes |

## ⚙️ What Each Service Does

### Frontend (React)
- Main user interface
- Calls Python backend for all operations
- Handles UI state and interactions

### Python Backend
- **Transcription**: Downloads YouTube videos and transcribes using Whisper AI
- **Summarization**: Provides multilingual summarization (50+ languages) using mBART
- **Keywords**: Extracts key topics from transcripts using KeyBERT
- **All processing**: Works completely locally - no API keys needed for transcription/summarization!

## 🔄 Service Integration

The frontend:
1. ✅ Uses Python backend for transcription (Whisper model - no API keys!)
2. ✅ Uses Python backend for summarization (mBART model + keywords)
3. ✅ Uses Gemini API for translation (optional - needs API key)
4. ✅ Uses Gemini API for chat features (optional - needs API key)

## 📝 Environment Variables Checklist

### Frontend (.env.local)
```env
VITE_API_KEY=your_gemini_api_key_here
```
**Note**: Only needed for translation and chat. Transcription and summarization work without it!

### Python Backend
No environment variables required (uses default port 8000)

## 🐳 Docker Alternative

If you prefer Docker for the Python backend:

```bash
docker build -f Dockerfile.python -t tubescribe-python-backend .
docker run -p 8000:8000 --memory="4g" tubescribe-python-backend
```

## ✅ Verification

1. **Frontend**: Open http://localhost:5173 - should see the app
2. **Node.js Backend**: http://localhost:7860 - should see "Backend Running"
3. **Python Backend**: http://localhost:8000/health - should return `{"status": "healthy"}`

## 🆘 Troubleshooting

### Python Backend Won't Start
- Ensure Python 3.10+ is installed
- Check that port 8000 is not in use
- Verify dependencies are installed: `pip install -r requirements.txt`

### Models Not Loading
- First run downloads ~2-3GB of models (be patient)
- Ensure stable internet connection
- Check available disk space (need ~5GB free)

### Port Conflicts
- Change ports in respective config files if needed
- Frontend: `vite.config.ts`
- Node.js: `server/index.js` (line 17)
- Python: `uvicorn app:app --port 8001` (or modify script)

## 📚 Additional Documentation

- **Main README**: [README.md](README.md)
- **Python Backend Details**: [PYTHON_BACKEND_README.md](PYTHON_BACKEND_README.md)

## 💡 Tips

1. **Development**: Run all three services in separate terminal windows
2. **Production**: Use process managers like PM2 or systemd
3. **Performance**: Python backend uses ~2-4GB RAM - ensure adequate resources
4. **First Run**: Python backend first request may be slower (model loading)

## 🎯 Minimal Setup (Gemini Only)

If you don't want to run the Python backend:
- Frontend will automatically use Gemini for all summarization
- You'll still get summaries, just without the keyword extraction feature
- No Python installation needed

This setup works perfectly fine - the Python backend is an enhancement, not a requirement!