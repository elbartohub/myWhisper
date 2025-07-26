#!/bin/bash
# Whisper Environment Setup
# Source this file to set up environment variables for using project-local Whisper models

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODELS_DIR="$SCRIPT_DIR/models"

# Set environment variables
export WHISPER_CACHE_DIR="$MODELS_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✓ Activated Python virtual environment (Python 3.12)"
else
    echo "⚠ Virtual environment not found at $SCRIPT_DIR/.venv"
fi

echo "✓ Whisper cache directory set to: $WHISPER_CACHE_DIR"
echo "✓ Environment ready for Whisper usage"
echo ""
echo "Available commands:"
echo "  python whisper_models.py status    # Check downloaded models"
echo "  python whisper_example.py          # Run example script"
echo "  whisper audio.wav                  # Transcribe audio file"
