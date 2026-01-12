<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# TubeScribe AI

An AI-powered video analysis tool that transcribes, summarizes, translates, and provides interactive chat about YouTube videos using Google Gemini API and multilingual AI models.

View your app in AI Studio: https://ai.studio/apps/drive/1XoeFMJu5ZwUmk2IaGEK5g7YAXiZo8_ht

## Features

- 🎥 **YouTube Transcription**: Extract transcripts using local Whisper AI model (no API keys needed!)
- ⏱️ **Real-time Progress**: Server-Sent Events (SSE) for live transcription progress updates
- 📝 **AI Summarization**: Generate summaries using Gemini AI or multilingual mBART models
- 🔑 **Keyword Extraction**: Automatically extract key topics and keywords (via Python backend)
- 🌍 **Multilingual Translation**: Translate content into 15+ languages
- 💬 **Interactive Chat**: Ask questions about video content with AI-powered responses
- 🎨 **Modern UI**: Beautiful, responsive interface with dark mode support
- 🚀 **GPU Acceleration**: Automatic GPU detection and utilization for faster processing

## Architecture

This project consists of two main components:

1. **Frontend** (React + TypeScript + Vite): User interface
2. **Python Backend** (`backend/app.py`): Provides video transcription, multilingual summarization, and keyword extraction

📚 **For detailed system architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Prerequisites

- **Node.js** 18+ (for frontend only)
- **Python** 3.10+ (required for backend - transcription, summarization, and keywords)
- **FFmpeg** (required for audio processing)
- **Gemini API Key** (optional - only needed for translation and chat features)

## Quick Start

Get TubeScribe AI up and running in minutes! Follow these steps:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Duzttt/TubeScribe-AI.git
cd tubescribe-ai
```

### Step 2: Set Up Python Backend (Required)

The Python backend handles transcription, summarization, and keyword extraction using local AI models (no API keys needed for these features).

**Check Prerequisites:**
- ✅ Python 3.10+ installed (`python --version`)
- ✅ FFmpeg installed (`ffmpeg -version`)
  - **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (RHEL/CentOS)

**Choose Your Setup Method:**

<details>
<summary><b>🚀 Option A: Automated Scripts (Recommended - Easiest)</b></summary>

**Windows (PowerShell):**
```powershell
.\scripts\start-python-backend.ps1
```

**Windows (CMD):**
```cmd
scripts\start-python-backend.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start-python-backend.sh
./scripts/start-python-backend.sh
```

**Note:** If you encounter permission errors on Windows:
```powershell
.\scripts\fix-permissions.ps1
```

</details>

<details>
<summary><b>🔧 Option B: Manual Setup</b></summary>

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (from project root)
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

</details>

<details>
<summary><b>🐳 Option C: Docker</b></summary>

```bash
# Build the image (from project root)
docker build -f backend/Dockerfile -t tubescribe-python-backend backend

# Run the container
docker run -p 8000:8000 --memory="4g" tubescribe-python-backend
```

</details>

**Verify Backend is Running:**
- Open `http://localhost:8000` in your browser
- You should see a JSON response with service info and model status
- Check `http://localhost:8000/health` - should return `{"status": "healthy"}`

**⏰ First Run:** The backend will automatically download AI models (~2-3GB) on first startup. This may take several minutes.

### Step 3: Set Up Frontend

**Install Dependencies:**
```bash
cd frontend
npm install
```

**Configure Environment Variables:**

Create a `.env.local` file in the `frontend` directory:

```env
VITE_API_KEY=your_gemini_api_key_here
```

> **Note:** The Gemini API key is optional. You need it only for:
> - Translation features
> - Interactive chat
> 
> **Transcription and summarization work without it** (uses local Python backend).

**Get a Gemini API Key (Optional):**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to `frontend/.env.local`

**Start Development Server:**
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Step 4: Verify Everything Works

1. **Backend Check:**
   - Visit `http://localhost:8000` - should show service info
   - Visit `http://localhost:8000/health` - should return `{"status": "healthy"}`

2. **Frontend Check:**
   - Open `http://localhost:5173` in your browser
   - You should see the TubeScribe AI interface

3. **Test Transcription:**
   - Paste a YouTube URL in the input field
   - Click "Transcribe"
   - Wait for the transcript to appear (first run may take longer as models load)

### What's Next?

- 📚 **Detailed Setup:** See [docs/PYTHON_BACKEND_README.md](docs/PYTHON_BACKEND_README.md) for advanced configuration
- 🚀 **GPU Setup:** See [docs/GPU_SETUP.md](docs/GPU_SETUP.md) to enable GPU acceleration
- 🔧 **Model Options:** See [docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md) for alternative AI models
- ❓ **Troubleshooting:** Check the [Troubleshooting](#troubleshooting) section below

## Environment Variables

### Frontend (frontend/.env.local)
```env
VITE_API_KEY=your_gemini_api_key_here
```

### Python Backend
The Python backend can be configured via environment variables (optional):
```env
PORT=8000
HOST=0.0.0.0
SUMMARIZATION_MODEL=facebook/mbart-large-50-many-to-many-mmt  # Optional: change summarization model
```

**Available Summarization Models:**
- `facebook/mbart-large-50-many-to-many-mmt` (default, multilingual, 50+ languages)
- `facebook/bart-large-cnn` (English only, faster, smaller)
- `google/pegasus-xsum` (English, abstractive summaries)
- `t5-base` or `t5-large` (General purpose, multilingual with proper tokenizer)

## How It Works

1. **Transcription**: 
   - Python backend downloads audio from YouTube using yt-dlp
   - Whisper AI model transcribes the audio locally (no API keys needed!)
   - Progress is tracked in real-time via Server-Sent Events (SSE)
   - Transcripts include timestamps for easy navigation

2. **Summarization**: 
   - Uses Python backend with mBART model for multilingual summarization (50+ languages)
   - KeyBERT extracts key topics and keywords from transcripts
   - All processing happens locally - no external API calls needed

3. **Translation**: Uses Gemini API to translate transcripts/summaries (optional - requires API key)

4. **Chat**: Uses Gemini API with video context for interactive Q&A (optional - requires API key)

## Project Structure

```
tubescribe-ai/
├── backend/                    # Python backend
│   ├── app.py                  # FastAPI application
│   ├── summary.py              # Summarization service
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker configuration
│   └── test_tubescribe.py      # Test suite
├── frontend/                   # React frontend
│   ├── components/             # React components
│   │   ├── ChatInterface.tsx
│   │   ├── KeywordsDisplay.tsx
│   │   ├── Menu.tsx
│   │   ├── ResultCard.tsx
│   │   └── YouTubeInput.tsx
│   ├── services/               # Frontend services
│   │   ├── geminiService.ts
│   │   └── summarizationService.ts
│   ├── App.tsx                 # Main app component
│   ├── index.tsx               # Entry point
│   ├── package.json            # Frontend dependencies
│   └── vite.config.ts          # Vite configuration
├── docs/                       # Documentation
│   ├── GPU_SETUP.md
│   ├── MODEL_OPTIONS.md
│   └── ... (other docs)
├── scripts/                    # Helper scripts
│   ├── start-python-backend.sh
│   └── start-python-backend.bat
└── README.md
```

## API Endpoints

### Python Backend

- `GET /` - Service info and model status
- `GET /health` - Health check endpoint
- `POST /api/transcribe` - Transcribe YouTube video (returns transcript with timestamps)
- `GET /api/progress/{video_id}` - Get transcription progress (polling endpoint)
- `GET /api/progress/{video_id}/stream` - Server-Sent Events (SSE) stream for real-time progress updates
- `POST /summarize` - Summarize text and extract keywords (multilingual support)

See [docs/PYTHON_BACKEND_README.md](docs/PYTHON_BACKEND_README.md) for detailed API documentation.


## Building for Production

```bash
# Build frontend
cd frontend
npm run build

# The built files will be in the frontend/dist/ directory
```

## Known Limitations

### Malay Language Support
**The models used in this project do not support Malay language.** While Malay may appear in some language lists, the underlying AI models (mBART-50, BART, and Whisper) do not provide reliable support for Malay transcription, summarization, or translation. Users processing Malay content may experience reduced accuracy, poor summarization results, or translation errors.

For more details, see [docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md#known-limitations).

## Troubleshooting

### Python Backend Issues
- See [docs/PYTHON_BACKEND_README.md](docs/PYTHON_BACKEND_README.md) for detailed troubleshooting
- Ensure you have at least 4GB RAM available (8GB+ recommended for optimal performance)
- First run will download ~2-3GB of model files (Whisper + mBART + KeyBERT)
- GPU setup: See [docs/GPU_SETUP.md](docs/GPU_SETUP.md) for CUDA installation and configuration
- Model options: See [docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md) for alternative model configurations

### Transcription Issues
- Ensure the Python backend is running on port 8000
- Check that FFmpeg is installed and accessible in your PATH
- Verify GPU availability with `python check_gpu.py` (GPU speeds up transcription significantly)
- Some videos may be unavailable or blocked (403 errors) - the backend will try multiple strategies automatically
- First transcription may take longer as models load into memory
- For long videos, transcription can take 1-2x the video duration on CPU, much faster on GPU

### Frontend Issues
- Clear browser cache
- Check browser console for errors
- Ensure all environment variables are set correctly

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.