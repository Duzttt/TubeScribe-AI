# Backend

This directory contains the Python backend for TubeScribe AI.

## Important Note

If `app.py` is missing, please restore it from version control:
```bash
git checkout HEAD -- app.py
```

Or if you have it in another location, copy it to this directory.

## Files

- `app.py` - FastAPI application (main backend server)
- `summary.py` - Summarization and keyword extraction service
- `requirements.txt` - Python dependencies
- `test_tubescribe.py` - Test suite
- `check_gpu.py` - GPU detection utility
- `Dockerfile` - Docker configuration
- `pytest.ini` - Pytest configuration

## Running the Backend

### Option 1: From Backend Directory (Easiest)

**Windows (PowerShell):**
```powershell
cd backend
.\start.ps1
```

**Windows (Command Prompt):**
```cmd
cd backend
start.bat
```

**Linux/Mac:**
```bash
cd backend
./start.sh
```

### Option 2: From Project Root

```bash
scripts/start-python-backend.sh    # Linux/Mac
scripts/start-python-backend.bat   # Windows
scripts/start-python-backend.ps1   # Windows PowerShell
```

### Option 3: Manual

```bash
# Activate venv first (from project root)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Then run from backend directory
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
