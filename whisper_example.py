#!/usr/bin/env python3
"""
Whisper Usage Example
This script demonstrates how to use Whisper with the custom model directory.
"""

import os
import whisper
from pathlib import Path

# Set the custom models directory
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

def setup_whisper_environment():
    """Set up the environment to use our custom models directory"""
    os.environ['WHISPER_CACHE_DIR'] = str(MODELS_DIR)
    print(f"Using models directory: {MODELS_DIR}")

def load_whisper_model(model_name="tiny"):
    """Load a Whisper model from our custom directory"""
    setup_whisper_environment()
    
    print(f"Loading {model_name} model...")
    try:
        # Load model with explicit download_root to ensure it uses our directory
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR))
        print(f"Successfully loaded {model_name} model from {MODELS_DIR}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def transcribe_audio(model, audio_file_path):
    """Transcribe an audio file using the loaded model"""
    if not model:
        print("No model loaded!")
        return None
    
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return None
    
    print(f"Transcribing: {audio_file_path}")
    try:
        result = model.transcribe(audio_file_path)
        # Save SRT file
        if result and "segments" in result:
            srt_path = Path(audio_file_path).with_suffix('.srt')
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = format_timestamp(segment['start'])
                    end = format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            print(f"SRT file saved to: {srt_path}")
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def example_usage():
    """Example of how to use Whisper with custom model directory"""
    print("=" * 60)
    print("Whisper Usage Example")
    print("=" * 60)
    
    # Load the model
    model = load_whisper_model("tiny")
    
    if model:
        print("\nModel loaded successfully!")
        print("You can now use it to transcribe audio files like this:")
        print("  result = transcribe_audio(model, 'path/to/your/audio.wav')")
        print("  print(result['text'])")
        
        # Show model info
        print(f"\nModel details:")
        print(f"  - Model name: tiny")
        print(f"  - Model location: {MODELS_DIR}/tiny.pt")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Ready to transcribe audio!")
    print("=" * 60)

if __name__ == "__main__":
    example_usage()
