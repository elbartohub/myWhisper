#!/usr/bin/env python3
"""
Whisper Model Manager
This script helps you download and manage Whisper models in your project directory.
"""

import os
import whisper
from pathlib import Path
import argparse

# Set up paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

# Model sizes and approximate download sizes
MODEL_INFO = {
    "tiny.en": {"size": "~39 MB", "description": "English-only, fastest"},
    "tiny": {"size": "~39 MB", "description": "Multilingual, fastest"},
    "base.en": {"size": "~74 MB", "description": "English-only, good balance"},
    "base": {"size": "~74 MB", "description": "Multilingual, good balance"},
    "small.en": {"size": "~244 MB", "description": "English-only, better accuracy"},
    "small": {"size": "~244 MB", "description": "Multilingual, better accuracy"},
    "medium.en": {"size": "~769 MB", "description": "English-only, high accuracy"},
    "medium": {"size": "~769 MB", "description": "Multilingual, high accuracy"},
    "large-v1": {"size": "~1550 MB", "description": "Multilingual, highest accuracy (v1)"},
    "large-v2": {"size": "~1550 MB", "description": "Multilingual, highest accuracy (v2)"},
    "large-v3": {"size": "~1550 MB", "description": "Multilingual, highest accuracy (v3)"},
    "large": {"size": "~1550 MB", "description": "Multilingual, highest accuracy (latest)"},
    "large-v3-turbo": {"size": "~809 MB", "description": "Multilingual, fast + accurate"},
    "turbo": {"size": "~809 MB", "description": "Multilingual, fast + accurate (latest)"}
}

def setup_environment():
    """Set up the environment for custom model directory"""
    MODELS_DIR.mkdir(exist_ok=True)
    os.environ['WHISPER_CACHE_DIR'] = str(MODELS_DIR)

def list_available_models():
    """List all available models with their details"""
    print("Available Whisper Models:")
    print("=" * 70)
    print(f"{'Model':<15} {'Size':<12} {'Description'}")
    print("-" * 70)
    
    for model_name in whisper.available_models():
        info = MODEL_INFO.get(model_name, {"size": "Unknown", "description": "Standard model"})
        print(f"{model_name:<15} {info['size']:<12} {info['description']}")

def list_downloaded_models():
    """List models that are already downloaded"""
    print(f"\nDownloaded Models in {MODELS_DIR}:")
    print("=" * 50)
    
    model_files = list(MODELS_DIR.glob("*.pt"))
    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_file.stem:<15} ({size_mb:.1f} MB)")
    else:
        print("  No models downloaded yet")

def download_model(model_name):
    """Download a specific model to the project directory"""
    setup_environment()
    
    if model_name not in whisper.available_models():
        print(f"Error: '{model_name}' is not a valid model name")
        list_available_models()
        return False
    
    model_file = MODELS_DIR / f"{model_name}.pt"
    if model_file.exists():
        print(f"Model '{model_name}' already exists in {MODELS_DIR}")
        return True
    
    print(f"Downloading {model_name} model...")
    info = MODEL_INFO.get(model_name, {})
    if 'size' in info:
        print(f"Expected download size: {info['size']}")
    
    try:
        # This will download the model to our custom directory
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR))
        print(f"✓ Successfully downloaded {model_name} model to {MODELS_DIR}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def remove_model(model_name):
    """Remove a downloaded model"""
    model_file = MODELS_DIR / f"{model_name}.pt"
    if model_file.exists():
        model_file.unlink()
        print(f"✓ Removed {model_name} model")
        return True
    else:
        print(f"Model '{model_name}' not found in {MODELS_DIR}")
        return False

def get_disk_usage():
    """Show disk usage of the models directory"""
    total_size = 0
    model_files = list(MODELS_DIR.glob("*.pt"))
    
    for model_file in model_files:
        total_size += model_file.stat().st_size
    
    total_mb = total_size / (1024 * 1024)
    total_gb = total_mb / 1024
    
    print(f"\nDisk Usage:")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Total size: {total_mb:.1f} MB ({total_gb:.2f} GB)")
    print(f"Number of models: {len(model_files)}")

def main():
    parser = argparse.ArgumentParser(description="Manage Whisper models in your project")
    parser.add_argument("action", choices=["list", "download", "remove", "status"], 
                       help="Action to perform")
    parser.add_argument("model", nargs="?", help="Model name (for download/remove actions)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_available_models()
    elif args.action == "download":
        if not args.model:
            print("Error: Please specify a model name to download")
            list_available_models()
        else:
            download_model(args.model)
    elif args.action == "remove":
        if not args.model:
            print("Error: Please specify a model name to remove")
            list_downloaded_models()
        else:
            remove_model(args.model)
    elif args.action == "status":
        list_downloaded_models()
        get_disk_usage()

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # No arguments provided, show interactive menu
        print("Whisper Model Manager")
        print("=" * 40)
        list_available_models()
        print()
        list_downloaded_models()
        get_disk_usage()
        print("\nUsage examples:")
        print("  python whisper_models.py list            # List available models")
        print("  python whisper_models.py download base   # Download base model")
        print("  python whisper_models.py remove tiny     # Remove tiny model")
        print("  python whisper_models.py status          # Show current status")
    else:
        main()
