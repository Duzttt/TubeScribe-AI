@echo off
REM Script to fix permission issues with Python virtual environment
REM Run this script as Administrator if the issue persists

echo Fixing Python virtual environment permissions...
echo.

REM Check if virtual environment exists
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to remove venv directory.
        echo.
        echo SOLUTION: Please close all Python processes and IDEs (VS Code, PyCharm, etc.)
        echo Then run this script again, or manually delete the venv folder.
        echo.
        pause
        exit /b 1
    )
) else (
    echo No existing virtual environment found.
)

echo.
echo Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
echo NOTE: If you still get permission errors, try:
echo 1. Close all Python processes and IDEs
echo 2. Run this script as Administrator (Right-click -^> Run as Administrator)
echo.

pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install.
    echo You may need to run this script as Administrator.
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Virtual environment created and dependencies installed!
    echo.
)

pause
