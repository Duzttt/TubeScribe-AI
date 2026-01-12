@echo off
REM Test runner script for Windows
echo Running TubeScribe AI Tests...
echo.

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo Installing test dependencies...
    pip install pytest pytest-asyncio httpx
    echo.
)

REM Run tests
echo Starting test suite...
python -m pytest test_tubescribe.py -v --tb=short

if errorlevel 1 (
    echo.
    echo Tests completed with failures.
    exit /b 1
) else (
    echo.
    echo All tests passed!
    exit /b 0
)
