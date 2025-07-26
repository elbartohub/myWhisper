#!/bin/bash
# Whisper Transcription Wrapper Script
# This script handles conda deactivation and runs the Python transcription script

echo "🎙️  Whisper Transcription Tool"
echo "================================"

# Deactivate conda if active
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "🔄 Deactivating conda environment: $CONDA_DEFAULT_ENV"
    eval "$(conda shell.bash hook)"
    conda deactivate
    echo "✓ Conda environment deactivated"
fi

# Check if virtual environment exists
if [[ ! -f ".venv/bin/python" ]]; then
    echo "❌ Virtual environment not found!"
    echo "Please run 'source setup_env.sh' first to set up the environment"
    exit 1
fi

# Activate Python virtual environment and set Whisper cache
source .venv/bin/activate
export WHISPER_CACHE_DIR="$(pwd)/models"

# Suppress MPS warnings for cleaner output
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "✓ Python virtual environment activated"
echo "✓ Whisper cache directory set to: $WHISPER_CACHE_DIR"
echo ""

# Run the Python transcription script with all passed arguments
python transcribe.py "$@"

# Store exit code
exit_code=$?

# Deactivate virtual environment
deactivate

echo ""
if [[ $exit_code -eq 0 ]]; then
    echo "🎉 Transcription completed successfully!"
else
    echo "❌ Transcription failed (exit code: $exit_code)"
fi

exit $exit_code
