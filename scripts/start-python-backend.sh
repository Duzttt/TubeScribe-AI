#!/bin/bash

# Script to start the Python summarization backend
# Usage: ./scripts/start-python-backend.sh

echo "🚀 Starting TubeScribe Python Summarization Backend..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import fastapi" &> /dev/null; then
    echo "📥 Installing dependencies (this may take several minutes on first run)..."
    pip install -r requirements.txt
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the project root."
    exit 1
fi

echo ""
echo "✅ Starting server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo "❤️  Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload