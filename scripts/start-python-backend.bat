@echo off
REM Script to start the Python summarization backend on Windows
REM Usage: scripts\start-python-backend.bat

echo Starting TubeScribe Python Summarization Backend...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python 3.10 or higher.
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment activation script not found.
    exit /b 1
)

REM Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (this may take several minutes on first run)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        exit /b 1
    )
)

REM Check if app.py exists
if not exist "app.py" (
    echo ERROR: app.py not found. Please run this script from the project root.
    exit /b 1
)

echo.
echo Starting server on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo Health check at http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload