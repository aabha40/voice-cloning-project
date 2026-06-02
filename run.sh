#!/bin/bash
# VoiceForge — Quick Start Script
# Authors: Aabha Shukla, Prachi Jha | CDAC Pune

echo "======================================"
echo "  VoiceForge — AI Voice Cloning"
echo "  CDAC Pune Internship Project"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "→ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "→ Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q

# Check GPU availability
python3 -c "import torch; print('✓ GPU available:', torch.cuda.is_available())"

echo ""
echo "→ Starting VoiceForge web server..."
echo "→ Open http://localhost:5000 in your browser"
echo ""

python3 app/voice_clone.py
