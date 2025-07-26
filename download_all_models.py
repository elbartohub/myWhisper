#!/usr/bin/env python3
"""
Download All Whisper Models
This script downloads all available Whisper models to your project directory.
"""

import os
import whisper
from pathlib import Path
import argparse
import time

# Set up paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

# Model sizes and approximate download sizes
ALL_MODELS = [
    "tiny.en", "tiny", "base.en", "base", 
    "small.en", "small", "medium.en", "medium", 
    "large-v1", "large-v2", "large-v3", "large", 
    "large-v3-turbo", "turbo"
]

# Model info for size estimation
MODEL_SIZES = {
    "tiny.en": 39, "tiny": 39, "base.en": 74, "base": 74,
    "small.en": 244, "small": 244, "medium.en": 769, "medium": 769,
    "large-v1": 1550, "large-v2": 1550, "large-v3": 1550, "large": 1550,
    "large-v3-turbo": 809, "turbo": 809
}

def setup_environment():
    """Set up the environment for custom model directory"""
    MODELS_DIR.mkdir(exist_ok=True)
    os.environ['WHISPER_CACHE_DIR'] = str(MODELS_DIR)

def get_downloaded_models():
    """Get list of already downloaded models"""
    return [f.stem for f in MODELS_DIR.glob("*.pt")]

def calculate_total_size(models_to_download):
    """Calculate total download size in MB"""
    return sum(MODEL_SIZES.get(model, 0) for model in models_to_download)

def download_model(model_name):
    """Download a specific model"""
    print(f"\n📥 Downloading {model_name} model...")
    size_mb = MODEL_SIZES.get(model_name, "Unknown")
    print(f"   Expected size: ~{size_mb} MB")
    
    try:
        start_time = time.time()
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR))
        elapsed = time.time() - start_time
        print(f"   ✅ Downloaded {model_name} in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"   ❌ Error downloading {model_name}: {e}")
        return False

def download_all_models(exclude_large=False, exclude_english_only=False):
    """Download all models with options to exclude certain types"""
    setup_environment()
    
    # Filter models based on options
    models_to_download = ALL_MODELS.copy()
    
    if exclude_large:
        # Exclude large models (>1GB)
        large_models = ["large-v1", "large-v2", "large-v3", "large"]
        models_to_download = [m for m in models_to_download if m not in large_models]
        print("🚫 Excluding large models (>1GB)")
    
    if exclude_english_only:
        # Exclude English-only models
        english_only = [m for m in models_to_download if m.endswith('.en')]
        models_to_download = [m for m in models_to_download if not m.endswith('.en')]
        print("🚫 Excluding English-only models")
    
    # Check what's already downloaded
    downloaded = get_downloaded_models()
    remaining = [m for m in models_to_download if m not in downloaded]
    
    if not remaining:
        print("🎉 All selected models are already downloaded!")
        return
    
    total_size_mb = calculate_total_size(remaining)
    total_size_gb = total_size_mb / 1024
    
    print(f"\n📊 Download Summary:")
    print(f"   Models to download: {len(remaining)}")
    print(f"   Total download size: ~{total_size_mb} MB ({total_size_gb:.1f} GB)")
    print(f"   Already downloaded: {len(downloaded)} models")
    
    # Ask for confirmation
    response = input(f"\n⚠️  This will download {total_size_gb:.1f} GB. Continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Download cancelled")
        return
    
    print(f"\n🚀 Starting download of {len(remaining)} models...")
    print("=" * 60)
    
    success_count = 0
    for i, model in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Processing {model}")
        if download_model(model):
            success_count += 1
        else:
            print(f"   ⚠️  Failed to download {model}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Download complete!")
    print(f"   Successfully downloaded: {success_count}/{len(remaining)} models")
    
    if success_count < len(remaining):
        failed = len(remaining) - success_count
        print(f"   ⚠️  {failed} models failed to download")

def download_recommended_set():
    """Download a recommended set of models for most use cases"""
    recommended = ["tiny", "base", "small", "turbo"]
    
    setup_environment()
    downloaded = get_downloaded_models()
    remaining = [m for m in recommended if m not in downloaded]
    
    if not remaining:
        print("🎉 All recommended models are already downloaded!")
        return
    
    total_size_mb = calculate_total_size(remaining)
    total_size_gb = total_size_mb / 1024
    
    print(f"📦 Recommended Model Set:")
    print(f"   - tiny: Fastest, good for testing")
    print(f"   - base: Good balance of speed/accuracy")
    print(f"   - small: Better accuracy, still fast")
    print(f"   - turbo: Latest, best speed/accuracy balance")
    print(f"\n📊 Total size: ~{total_size_mb} MB ({total_size_gb:.1f} GB)")
    
    response = input(f"\n⚠️  Download recommended set? [Y/n]: ").strip().lower()
    if response in ['', 'y', 'yes']:
        for model in remaining:
            download_model(model)

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--recommended", action="store_true", help="Download recommended set")
    parser.add_argument("--exclude-large", action="store_true", help="Exclude large models (>1GB)")
    parser.add_argument("--exclude-english-only", action="store_true", help="Exclude English-only models")
    
    args = parser.parse_args()
    
    if args.all:
        download_all_models(
            exclude_large=args.exclude_large,
            exclude_english_only=args.exclude_english_only
        )
    elif args.recommended:
        download_recommended_set()
    else:
        # Interactive mode
        print("🎯 Whisper Model Downloader")
        print("=" * 40)
        print("1. Download all models (~8.5 GB)")
        print("2. Download recommended set (~1.2 GB)")
        print("3. Download all except large models (~3.3 GB)")
        print("4. Exit")
        
        choice = input("\nSelect option [1-4]: ").strip()
        
        if choice == "1":
            download_all_models()
        elif choice == "2":
            download_recommended_set()
        elif choice == "3":
            download_all_models(exclude_large=True)
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
