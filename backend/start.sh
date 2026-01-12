#!/bin/bash

# Script to start the backend from backend directory
# Usage: ./start.sh (from backend directory)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$BACKEND_DIR")"

echo "🚀 Starting TubeScribe Python Backend..."
echo ""

# Change to project root for venv access
cd "$PROJECT_ROOT"

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
    pip install -r "$BACKEND_DIR/requirements.txt"
fi

# Check if app.py exists
if [ ! -f "$BACKEND_DIR/app.py" ]; then
    echo "❌ app.py not found in backend directory."
    exit 1
fi

echo ""
echo "✅ Starting server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo "❤️  Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server from backend directory
cd "$BACKEND_DIR"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
