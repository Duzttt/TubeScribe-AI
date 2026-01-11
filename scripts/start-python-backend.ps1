# PowerShell script to start the Python summarization backend on Windows
# Usage: .\scripts\start-python-backend.ps1

Write-Host "Starting TubeScribe Python Summarization Backend..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "ERROR: Virtual environment activation script not found." -ForegroundColor Red
    Write-Host "Trying alternative activation method..." -ForegroundColor Yellow
    $env:VIRTUAL_ENV = (Resolve-Path "venv").Path
    $env:PATH = "$env:VIRTUAL_ENV\Scripts;$env:PATH"
}

# Check if requirements are installed
try {
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw
    }
} catch {
    Write-Host "Installing dependencies (this may take several minutes on first run)..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
}

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "API docs available at http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Health check at http://localhost:8000/health" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload