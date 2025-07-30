"""
Whisper Transcription Script
Handles environment setup and transcribes audio/video files using Whisper models.
"""

# Suppress PyTorch backend registration logs by redirecting stderr before any imports
import sys
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass
sys.stderr = DevNull()

import os
import subprocess
import argparse
import warnings
from pathlib import Path
import whisper
# ...existing code...
import os
import subprocess
import argparse
import warnings
from pathlib import Path
import whisper
import json
import torch
print("[CUDA] transcribe.py: CUDA available:", torch.cuda.is_available())
# ...existing code...
# Set up paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
VENV_DIR = PROJECT_DIR / ".venv"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.wav', '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', 
    '.m4a', '.aac', '.ogg', '.flac', '.wma', '.3gp', '.amr'
}

def get_device():
    """Detect the best available device for Whisper"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def print_device_info():
    """Print information about the device being used"""
    device = get_device()
    if device == "mps":
        print("🚀 Using Apple Silicon GPU (MPS) for faster transcription")
        print("   Note: Some operations may fall back to CPU (this is normal)")
    elif device == "cuda":
        print("🚀 Using NVIDIA GPU (CUDA) for faster transcription")
    else:
        print("💻 Using CPU for transcription")
    return device

def setup_environment():
    """Set up the environment for Whisper"""
    # Suppress MPS warnings and logs
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Deactivate conda if active
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print("🔄 Deactivating current conda environment...")
        try:
            subprocess.run(['conda', 'deactivate'], shell=True, check=False)
            print("✓ Conda environment deactivated")
        except Exception as e:
            print(f"⚠️  Could not deactivate conda: {e}")
    
    # Activate Python virtual environment
    if sys.platform == "win32":
        activate_script = VENV_DIR / "Scripts" / "activate"
        python_executable = VENV_DIR / "Scripts" / "python"
    else:
        activate_script = VENV_DIR / "bin" / "activate"
        python_executable = VENV_DIR / "bin" / "python"
    
    if not python_executable.exists():
        print(f"❌ Virtual environment not found at {VENV_DIR}")
        print("Please run 'source setup_env.sh' first to set up the environment")
        return False
    
    # Set environment variables
    os.environ['VIRTUAL_ENV'] = str(VENV_DIR)
    os.environ['PATH'] = f"{VENV_DIR / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ['WHISPER_CACHE_DIR'] = str(MODELS_DIR)
    
    print("✓ Python virtual environment activated")
    print(f"✓ Whisper cache directory set to: {MODELS_DIR}")
    
    return True

def check_file_exists(file_path):
    """Check if the provided file exists and is supported"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"❌ Unsupported file format: {path.suffix}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return False
    
    return True

def list_available_models():
    """List downloaded models"""
    model_files = list(MODELS_DIR.glob("*.pt"))
    if not model_files:
        print("❌ No models found. Please download a model first:")
        print("   python whisper_models.py download base")
        return []
    
    models = []
    print("\n📁 Available models:")
    for i, model_file in enumerate(model_files, 1):
        model_name = model_file.stem
        size_mb = model_file.stat().st_size / (1024 * 1024)
        models.append(model_name)
        print(f"   {i}. {model_name} ({size_mb:.1f} MB)")
    
    return models

def choose_model(models, specified_model=None):
    """Let user choose a model or use the specified one"""
    if specified_model:
        if specified_model in models:
            return specified_model
        else:
            print(f"❌ Model '{specified_model}' not found locally")
            print("Available models:", ", ".join(models))
            return None
    
    # Interactive model selection
    while True:
        try:
            print(f"\n🎯 Choose a model (1-{len(models)}) or press Enter for 'base':")
            choice = input("Model choice: ").strip()
            
            if not choice:
                return 'base' if 'base' in models else models[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                return models[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            # Maybe they entered model name directly
            if choice in models:
                return choice
            print("Please enter a valid number or model name")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return None


def transcribe_file(file_path, model_name, output_dir=None, output_format="txt", translate_zh=False):
    """Transcribe the audio/video file, with optional Traditional Chinese translation"""
    print(f"\n🎤 Loading Whisper model: {model_name}")

    # Detect and display device info
    device = print_device_info()

    try:
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR), device=device)
    except Exception:
        # Hide detailed error, just show fallback message
        if device != "cpu":
            print("⚠️  Could not load model on GPU. Falling back to CPU...")
            try:
                model = whisper.load_model(model_name, download_root=str(MODELS_DIR), device="cpu")
                device = "cpu"
                print("💻 Using CPU for transcription")
            except Exception:
                print("❌ Error loading model on CPU.")
                return False
        else:
            print("❌ Error loading model.")
            return False

    print(f"🎵 Transcribing: {Path(file_path).name}")
    print("⏳ This may take a while depending on file length and model size...")

    # Show file info
    file_size = Path(file_path).stat().st_size / (1024 * 1024)
    print(f"📊 File size: {file_size:.1f} MB")

    try:
        # Suppress all warnings during transcription for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Additional environment setup for cleaner MPS output
            import logging
            logging.getLogger().setLevel(logging.ERROR)

            # Use fp16 only if using GPU (MPS or CUDA)
            fp16 = device in ["mps", "cuda"]
            result = model.transcribe(str(file_path), verbose=False, fp16=fp16)

        # If translation to Traditional Chinese is requested
        if translate_zh:
            import time
            import re
            print("[INFO] Entering translation block for Traditional Chinese...")
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='auto', target='zh-TW')
                # Translate main text (split into sentences)
                sentences = re.split(r'(?<=[.!?。！？])\s+', result["text"])
                zh_sentences = []
                for s in sentences:
                    if s.strip():
                        try:
                            zh = translator.translate(s.strip())
                            zh_sentences.append(zh)
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"[ERROR] Main text sentence translation failed: {e}\nOriginal: {s}")
                zh_text = ' '.join(zh_sentences)
                result["text"] = zh_text
                print(f"[DEBUG] Main text translated: {zh_text[:40]}...")
                # Translate each segment (split into sentences)
                for idx, seg in enumerate(result.get("segments", [])):
                    original = seg["text"]
                    seg_sentences = re.split(r'(?<=[.!?。！？])\s+', original)
                    zh_seg_sentences = []
                    for s in seg_sentences:
                        if s.strip():
                            try:
                                zh = translator.translate(s.strip())
                                zh_seg_sentences.append(zh)
                                time.sleep(0.5)
                            except Exception as seg_e:
                                print(f"[ERROR] Segment {idx+1} sentence translation failed: {seg_e}\nOriginal: {s}")
                    translated = ' '.join(zh_seg_sentences)
                    seg["text"] = translated
                    print(f"[DEBUG] Segment {idx+1} translated: {translated}")
                print("✅ Translated output to Traditional Chinese (zh-TW) using deep-translator (sentence by sentence)")
            except Exception as e:
                print(f"⚠️  Translation error: {e}\nIf you have not installed deep-translator, run: pip install deep-translator")

        # Prepare output
        input_path = Path(file_path)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_base = output_dir / input_path.stem
        else:
            output_base = input_path.parent / input_path.stem

        # Save transcription in requested format
        if output_format in ["txt", "all"]:
            txt_file = output_base.with_suffix('.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result["text"].strip())
            print(f"✅ Text saved to: {txt_file}")

        if output_format in ["json", "all"]:
            json_file = output_base.with_suffix('.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON saved to: {json_file}")

        if output_format in ["srt", "all"]:
            srt_file = output_base.with_suffix('.srt')
            write_srt(result["segments"], srt_file)
            print(f"✅ SRT saved to: {srt_file}")

        # Show transcription stats
        if "segments" in result:
            duration = result.get("duration", 0)
            num_segments = len(result["segments"])
            print(f"\n📈 Transcription stats:")
            print(f"   Device used: {device.upper()}")
            print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"   Segments: {num_segments}")
            print(f"   Words: ~{len(result['text'].split())}")

        # Display result
        print(f"\n📝 Transcription:")
        print("=" * 60)
        print(result["text"].strip())
        print("=" * 60)

        # Full SRT preview in terminal by default
        if output_format in ["srt", "all"]:
            srt_file = output_base.with_suffix('.srt')
            if srt_file.exists():
                print(f"\n🔎 Full SRT Preview: {srt_file}")
                print("-" * 60)
                with open(srt_file, 'r', encoding='utf-8') as f:
                    print(f.read())
                print("-" * 60)
                print("\n✅ Preview complete.")

        return True
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return False

def write_srt(segments, output_file):
    """Write segments to SRT subtitle format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files with Whisper")
    parser.add_argument("file_path", nargs="?", help="Path to audio/video file")
    parser.add_argument("-m", "--model", help="Whisper model to use")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-f", "--format", choices=["txt", "json", "srt", "all"], default="srt", help="Output format (default: srt)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (disable MPS/CUDA)")
    parser.add_argument("--preview-srt", action="store_true", help="Preview full SRT in terminal and skip transcription")
    parser.add_argument("--translate-zh", action="store_true", help="Translate output to Traditional Chinese (zh-TW)")


    args = parser.parse_args()

    # Override device detection if --cpu flag is used
    if args.cpu:
        global get_device
        def get_device():
            return "cpu"

    # Suppress PyTorch backend registration warnings by redirecting stderr
    import sys
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    class DevNull:
        def write(self, msg):
            pass
        def flush(self):
            pass
    sys.stderr = DevNull()

    print("🎙️  Whisper Transcription Tool")
    print("=" * 50)

    # Get file path
    if args.file_path:
        file_path = args.file_path
    else:
        print("\n📁 Enter the path to your audio/video file:")
        file_path = input("File path: ").strip().strip('"\'')
        if not file_path:
            print("❌ No file path provided")
            return 1

    # If preview-srt is requested, show full SRT and exit
    if args.preview_srt:
        srt_path = Path(file_path).with_suffix('.srt')
        if not srt_path.exists():
            print(f"❌ SRT file not found: {srt_path}")
            return 1
        print(f"\n🔎 Full SRT Preview: {srt_path}")
        print("-" * 60)
        with open(srt_path, 'r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 60)
        print("\n✅ Preview complete.")
        return 0

    # Setup environment
    if not setup_environment():
        return 1

    # Check file
    if not check_file_exists(file_path):
        return 1

    # List and choose model
    models = list_available_models()
    if not models:
        return 1

    model_name = choose_model(models, args.model)
    if not model_name:
        return 1

    # Transcribe
    success = transcribe_file(file_path, model_name, args.output, args.format, args.translate_zh)

    if success:
        print("\n🎉 Transcription completed successfully!")
        return 0
    else:
        print("\n❌ Transcription failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
