@echo off
REM Script to start the backend from backend directory
REM Usage: start.bat (from backend directory)

cd /d %~dp0
set BACKEND_DIR=%~dp0
cd /d %~dp0\..

echo Starting TubeScribe Python Backend...
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
    pip install -r "%BACKEND_DIR%requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        exit /b 1
    )
)

REM Check if app.py exists
if not exist "%BACKEND_DIR%app.py" (
    echo ERROR: app.py not found in backend directory.
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
cd /d %BACKEND_DIR%
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
