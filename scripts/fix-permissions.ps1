# PowerShell script to fix permission issues with Python virtual environment
# Run as Administrator if needed: Right-click PowerShell -> Run as Administrator

Write-Host "Fixing Python virtual environment permissions..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    
    # Try to remove venv
    try {
        Remove-Item -Path "venv" -Recurse -Force -ErrorAction Stop
        Write-Host "Virtual environment removed successfully." -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to remove venv directory." -ForegroundColor Red
        Write-Host ""
        Write-Host "SOLUTION: Please close all Python processes and IDEs:" -ForegroundColor Yellow
        Write-Host "1. Close VS Code, PyCharm, or any other IDE" -ForegroundColor Yellow
        Write-Host "2. Close all Python processes in Task Manager" -ForegroundColor Yellow
        Write-Host "3. Run this script again, or manually delete the venv folder" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "If the issue persists, run PowerShell as Administrator:" -ForegroundColor Yellow
        Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
        Write-Host ""
        
        # Try to kill Python processes
        $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
        if ($pythonProcesses) {
            Write-Host "Found Python processes running. Attempting to close them..." -ForegroundColor Yellow
            $pythonProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
            
            # Try again
            try {
                Remove-Item -Path "venv" -Recurse -Force -ErrorAction Stop
                Write-Host "Virtual environment removed after closing processes." -ForegroundColor Green
            } catch {
                Write-Host "ERROR: Still cannot remove. Please close processes manually and try again." -ForegroundColor Red
                exit 1
            }
        } else {
            exit 1
        }
    }
} else {
    Write-Host "No existing virtual environment found." -ForegroundColor Green
}

Write-Host ""
Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Check if activation script exists (might need to set execution policy)
if (-not $?) {
    Write-Host "Activation script blocked. Setting execution policy..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
    & "venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host ""
Write-Host "Installing dependencies (this may take several minutes)..." -ForegroundColor Yellow
Write-Host "NOTE: If you still get permission errors:" -ForegroundColor Yellow
Write-Host "1. Close all Python processes and IDEs" -ForegroundColor Yellow
Write-Host "2. Run PowerShell as Administrator" -ForegroundColor Yellow
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Some packages failed to install." -ForegroundColor Red
    Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host ""
    Write-Host "SUCCESS: Virtual environment created and dependencies installed!" -ForegroundColor Green
    Write-Host ""
}
