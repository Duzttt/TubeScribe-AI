# Python Backend

This is the Python FastAPI backend service that provides video transcription, multilingual text summarization, and keyword extraction capabilities.

## Features

- **Video Transcription**: Uses Whisper model for automatic speech recognition (no API keys needed!)
- **Multilingual Summarization**: Uses mBART model supporting 50+ languages
- **Keyword Extraction**: Uses KeyBERT with multilingual embeddings
- **RESTful API**: FastAPI endpoints for easy integration
- **Error Handling**: Robust error handling and health checks

## Prerequisites

- Python 3.10 or higher
- pip package manager
- **FFmpeg** (required for audio processing)
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (RHEL/CentOS)
  - **Mac**: `brew install ffmpeg`

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The first installation will download large model files (~2-3 GB). This may take several minutes.

2. Verify installation:
   ```bash
   python -c "import transformers; import keybert; print('Dependencies installed successfully!')"
   ```

## Running the Backend

### Option 1: Using uvicorn (Development)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload on code changes (useful for development).

### Option 2: Using Docker

1. Build the Docker image:
   ```bash
   docker build -f Dockerfile.python -t tubescribe-python-backend .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 tubescribe-python-backend
   ```

   Or with additional memory allocation (recommended for larger models):
   ```bash
   docker run -p 8000:8000 --memory="4g" tubescribe-python-backend
   ```

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service and model availability.

**Response:**
```json
{
  "status": "healthy",
  "models_ready": true
}
```

### Root Endpoint
```
GET /
```
Returns service information and model status.

### Transcribe Video
```
POST /api/transcribe
```

**Request Body:**
```json
{
  "videoUrl": "https://www.youtube.com/watch?v=...",
  "targetLanguage": null  // Optional, for translation (not yet implemented in transcription)
}
```

**Response:**
```json
{
  "transcript": "[00:00] First line of transcript...\n[00:05] Second line..."
}
```

**Note**: Transcription uses Whisper model locally - no external API keys needed! First transcription may take longer as the model processes the audio.

### Summarize
```
POST /summarize
```

**Request Body:**
```json
{
  "text": "Your text to summarize here...",
  "lang": "en_XX"  // Optional, defaults to "en_XX"
}
```

**Response:**
```json
{
  "summary": "Summarized text here...",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

**Supported Language Codes:**
- `en_XX` - English
- `zh_CN` - Chinese (Simplified)
- `es_XX` - Spanish
- `fr_XX` - French
- `de_DE` - German
- `ja_XX` - Japanese
- `ko_KR` - Korean
- `hi_IN` - Hindi
- `ar_AR` - Arabic
- `pt_XX` - Portuguese
- `it_IT` - Italian
- `ru_RU` - Russian
- And 40+ more languages supported by mBART

## Configuration

### Environment Variables

You can configure the backend using environment variables:

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Changing the Backend URL

In the frontend, the backend URL is configured in `services/summarizationService.ts`:

```typescript
const SUMMARIZATION_BACKEND_URL = process.env.SUMMARIZATION_BACKEND_URL || "http://localhost:8000";
```

To use a different URL, set the `SUMMARIZATION_BACKEND_URL` environment variable in your frontend build process.

## Performance Notes

- **First Request**: The first request may be slower as models are loaded into memory
- **Memory Usage**: The backend uses approximately 3-6 GB of RAM (Whisper model + summarization models)
- **CPU vs GPU**: Currently configured for CPU. For faster transcription, consider using GPU with CUDA support
- **Transcription Speed**: ~1-2x real-time on CPU (a 5-minute video takes ~5-10 minutes to transcribe)
- **Chunking**: Long texts (>500 words) are automatically chunked for processing
- **Audio Download**: Videos are downloaded and processed - ensure sufficient disk space for temporary files

## Troubleshooting

### FFmpeg Not Found
If you get errors about FFmpeg:
1. Install FFmpeg (see Prerequisites above)
2. Ensure FFmpeg is in your system PATH
3. Verify: `ffmpeg -version` should work in terminal

### Model Download Issues
If models fail to download:
1. Check your internet connection
2. Ensure you have at least 8 GB free disk space (for Whisper + summarization models)
3. Try running: `python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base')"`

### Windows Permission Errors (`[WinError 5] Access is denied`)

If you get permission errors when installing packages:

**Quick Fix:**
1. Close all Python processes, IDEs (VS Code, PyCharm), and terminals
2. Delete the `venv` folder manually
3. Run the fix script:
   ```powershell
   # PowerShell (Recommended)
   .\scripts\fix-permissions.ps1
   
   # Or Command Prompt
   scripts\fix-permissions.bat
   ```

**If the issue persists:**
1. Run PowerShell as Administrator (Right-click → Run as Administrator)
2. Navigate to project directory: `cd path\to\tubescribe-ai`
3. Run the fix script again

**Manual Fix:**
```powershell
# Kill all Python processes
Get-Process python | Stop-Process -Force

# Remove venv
Remove-Item -Path venv -Recurse -Force

# Recreate venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Memory Errors
If you encounter out-of-memory errors:
- Reduce `chunk_size` in `app.py` (default: 500)
- Use a machine with more RAM (4GB+ recommended)
- Consider using GPU acceleration

### Port Already in Use
If port 8000 is already in use:
```bash
# Change port in the command
uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# Or kill the process using port 8000 (Linux/Mac)
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Integration with Frontend

The frontend automatically detects if the Python backend is available. If the backend is running:
- **Transcription** will use the Python backend with Whisper (no API keys needed!)
- **Summarization** will use the Python backend (faster, includes keywords)
- If the backend is unavailable, it falls back to Node.js backend (for transcription) or Gemini API (for summarization)

The system is designed to work without any API keys when using the Python backend!

## License

Same as the main TubeScribe AI project.