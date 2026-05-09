#!/bin/bash
# quick_setup.sh - One-command setup after downloading model

echo "🎽 Clothing Classifier - Quick Setup"
echo "===================================="

# Check if model exists
if [ ! -f "ai_model/best_model.pth" ]; then
    echo "❌ Error: Model not found at ai_model/best_model.pth"
    echo "   Please place best_model.pth in ai_model/ folder"
    exit 1
fi

echo "✅ Model found: ai_model/best_model.pth"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python3 found"

# Create venv if needed
if [ ! -d "web_app/venv" ]; then
    echo "📦 Creating virtual environment..."
    cd web_app
    python3 -m venv venv
    source venv/bin/activate
    cd ..
else
    echo "✅ Virtual environment exists"
    source web_app/venv/bin/activate
fi

# Install requirements
echo "📦 Installing dependencies..."
pip install -r web_app/requirements_web.txt > /dev/null 2>&1

echo "✅ Dependencies installed"

# Print instructions
echo ""
echo "===================================="
echo "✨ Setup Complete!"
echo "===================================="
echo ""
echo "🚀 To start the web app, run:"
echo "   cd web_app"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🌐 Then open: http://localhost:8000"
echo ""
