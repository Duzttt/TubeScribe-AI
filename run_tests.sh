#!/bin/bash
# Test runner script for Linux/Mac

echo "Running TubeScribe AI Tests..."
echo ""

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    echo "Installing test dependencies..."
    pip install pytest pytest-asyncio httpx
    echo ""
fi

# Run tests
echo "Starting test suite..."
python -m pytest test_tubescribe.py -v --tb=short

# Exit with pytest's exit code
exit $?
