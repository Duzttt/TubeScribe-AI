# PowerShell script to start the backend from backend directory
# Usage: .\start.ps1 (from backend directory)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = $scriptDir
$projectRoot = Split-Path -Parent $backendDir

Write-Host "Starting TubeScribe Python Backend..." -ForegroundColor Cyan
Write-Host ""

# Change to project root for venv access
Set-Location $projectRoot

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
$venvPath = Join-Path $projectRoot "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "Trying alternative activation method..." -ForegroundColor Yellow
    $env:VIRTUAL_ENV = $venvPath
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
    pip install -r (Join-Path $backendDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
}

# Check if app.py exists
$appPath = Join-Path $backendDir "app.py"
if (-not (Test-Path $appPath)) {
    Write-Host "ERROR: app.py not found in backend directory." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "API docs available at http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Health check at http://localhost:8000/health" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server from backend directory
Set-Location $backendDir
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
