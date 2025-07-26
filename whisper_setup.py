#!/usr/bin/env python3
"""
Whisper Setup Script
This script configures Whisper to use the project's models directory
instead of the default system cache directory.
"""

import os
import whisper
import shutil
from pathlib import Path

# Set up paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "whisper"

def setup_model_directory():
    """Create the models directory if it doesn't exist"""
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")
    return MODELS_DIR

def copy_existing_models():
    """Copy any existing models from the default cache to our project directory"""
    if DEFAULT_CACHE_DIR.exists():
        print(f"Found existing Whisper cache at: {DEFAULT_CACHE_DIR}")
        
        for model_file in DEFAULT_CACHE_DIR.glob("*.pt"):
            target_file = MODELS_DIR / model_file.name
            if not target_file.exists():
                print(f"Copying {model_file.name} to project models directory...")
                shutil.copy2(model_file, target_file)
            else:
                print(f"Model {model_file.name} already exists in project directory")
    else:
        print("No existing Whisper cache found")

def set_whisper_cache_dir():
    """Set the environment variable to use our project models directory"""
    os.environ['WHISPER_CACHE_DIR'] = str(MODELS_DIR)
    print(f"Set WHISPER_CACHE_DIR to: {MODELS_DIR}")

def load_model_with_custom_path(model_name="tiny"):
    """Load a Whisper model using the custom path"""
    set_whisper_cache_dir()
    
    print(f"Loading {model_name} model...")
    print(f"Models will be stored in: {MODELS_DIR}")
    
    try:
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR))
        print(f"Successfully loaded {model_name} model!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def list_available_models():
    """List all available Whisper models"""
    print("Available Whisper models:")
    for model in whisper.available_models():
        print(f"  - {model}")

def list_downloaded_models():
    """List models that are already downloaded in our project directory"""
    print(f"\nModels in project directory ({MODELS_DIR}):")
    model_files = list(MODELS_DIR.glob("*.pt"))
    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
    else:
        print("  No models downloaded yet")

if __name__ == "__main__":
    print("=" * 50)
    print("Whisper Project Setup")
    print("=" * 50)
    
    # Setup
    setup_model_directory()
    copy_existing_models()
    
    # Show current status
    list_available_models()
    list_downloaded_models()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print(f"Your Whisper models will be stored in: {MODELS_DIR}")
    print("\nTo use Whisper with this setup:")
    print("1. Run this script to ensure models are in the right place")
    print("2. Use the load_model_with_custom_path() function in your code")
    print("3. Or set WHISPER_CACHE_DIR environment variable before importing whisper")
