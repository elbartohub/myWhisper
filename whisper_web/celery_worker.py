
# threading-based background transcription worker
import threading
import os
import subprocess
import time
import uuid
from pydub import AudioSegment, silence

# Global dictionary to track job progress and results
transcription_jobs = {}

def transcribe_task(job_id, file_path, output_dir, model, fmt, cpu, translate_zh):
    print(f"[DEBUG] Thread started for job_id={job_id}")
    print(f"[DEBUG] file_path={file_path}, output_dir={output_dir}, model={model}, fmt={fmt}, cpu={cpu}, translate_zh={translate_zh}")
    import datetime
    start_time = datetime.datetime.now().isoformat()
    transcription_jobs[job_id] = {
        'state': 'STARTED',
        'progress': 0,
        'stage': 'chunking',
        'chunk_progress': 0,
        'transcribe_progress': 0,
        'translate_progress': 0,
        'post_progress': 0,
        'start_time': start_time
    }
    import traceback
    try:
        import whisper
        import torch
        import warnings
        import re
        device = 'cpu'  # Always use CPU to avoid MPS backend errors
        transcription_jobs[job_id].update({'state': 'PROGRESS', 'progress': 5})
        model_name = model or 'base'
        print(f"[DEBUG] Loading Whisper model: {model_name} on device: {device}")
        model_obj = whisper.load_model(model_name, device=device)
        print(f"[DEBUG] Model loaded successfully.")

        # --- Chunking logic ---
        print(f"[DEBUG] Splitting audio into chunks...")
        transcription_jobs[job_id].update({'stage': 'chunking', 'chunk_progress': 0})
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = 60 * 1000  # 60 seconds
        silence_thresh = audio.dBFS - 16  # threshold for silence
        min_silence_len = 700  # ms
        chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=300)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_length_ms:
                for i in range(0, len(chunk), chunk_length_ms):
                    final_chunks.append(chunk[i:i+chunk_length_ms])
            else:
                final_chunks.append(chunk)
        if not final_chunks:
            final_chunks = [audio]
        print(f"[DEBUG] Total chunks: {len(final_chunks)}")
        transcription_jobs[job_id].update({'chunk_progress': 100, 'progress': 25})

        # Save temp chunk files
        temp_dir = os.path.join(os.path.dirname(file_path), f"_chunks_{job_id}")
        os.makedirs(temp_dir, exist_ok=True)
        chunk_files = []
        for idx, chunk in enumerate(final_chunks):
            chunk_path = os.path.join(temp_dir, f"chunk_{idx}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_files.append(chunk_path)

        # Transcribe each chunk and update progress
        transcription_jobs[job_id].update({'stage': 'transcribing', 'transcribe_progress': 0})
        all_segments = []
        all_text = []
        for idx, chunk_path in enumerate(chunk_files):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp16 = device in ["mps", "cuda"]
                print(f"[DEBUG] Transcribing chunk {idx+1}/{len(chunk_files)}: {chunk_path}")
                result = model_obj.transcribe(chunk_path, verbose=False, fp16=fp16)
                print(f"[DEBUG] Chunk {idx+1} transcription complete.")
            # Collect segments and text
            all_segments.extend(result.get("segments", []))
            all_text.append(result["text"].strip())
            percent = int(100 * (idx + 1) / len(chunk_files))
            transcription_jobs[job_id].update({'transcribe_progress': percent})
        transcription_jobs[job_id].update({'transcribe_progress': 100, 'progress': 50})
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Merge results
        result = {"segments": all_segments, "text": "\n".join(all_text)}
        if translate_zh:
            transcription_jobs[job_id].update({'stage': 'translating', 'translate_progress': 0})
            print(f"[DEBUG] Starting translation to Traditional Chinese...")
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='auto', target='zh-TW')
                sentences = re.split(r'(?<=[.!?。！？])\s+', result["text"])
                zh_sentences = []
                for idx, s in enumerate(sentences):
                    if s.strip():
                        zh = translator.translate(s.strip())
                        zh_sentences.append(zh)
                        time.sleep(0.5)
                    percent = int(100 * (idx + 1) / len(sentences))
                    transcription_jobs[job_id].update({'translate_progress': percent})
                zh_text = ' '.join(zh_sentences)
                result["text"] = zh_text
                print(f"[DEBUG] Main text translated.")
                for seg in result.get("segments", []):
                    seg_sentences = re.split(r'(?<=[.!?。！？])\s+', seg["text"])
                    zh_seg_sentences = []
                    for s in seg_sentences:
                        if s.strip():
                            zh = translator.translate(s.strip())
                            zh_seg_sentences.append(zh)
                            time.sleep(0.5)
                    seg["text"] = ' '.join(zh_seg_sentences)
                print(f"[DEBUG] Segments translated.")
            except Exception as e:
                print(f"[ERROR] Translation error: {e}")
                result["text"] += f"\n[Translation Error: {e}]"
            transcription_jobs[job_id].update({'translate_progress': 100, 'progress': 75})
        transcription_jobs[job_id].update({'stage': 'postprocessing', 'post_progress': 50})
        time.sleep(0.5)
        transcription_jobs[job_id].update({'post_progress': 100, 'progress': 100})
        time.sleep(0.5)
        output_text = result["text"].strip()
        output_file_path = None
        print(f"[DEBUG] Transcription output prepared.")
        custom_dict_path = os.path.join(os.path.dirname(__file__), 'custom_dict.txt')
        replacements = []
        if os.path.exists(custom_dict_path):
            print(f"[DEBUG] Applying custom dictionary replacements from {custom_dict_path}")
            try:
                with open(custom_dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            src, tgt = line.split('=', 1)
                            src = src.strip()
                            tgt = tgt.strip()
                            replacements.append((src, tgt))
                print(f"[DEBUG] Custom dictionary loaded: {replacements}")
            except Exception as e:
                print(f"[ERROR] Custom Dictionary Error: {e}")
        # Apply replacements to output_text
        for src, tgt in replacements:
            output_text = re.sub(re.escape(src), tgt, output_text, flags=re.IGNORECASE)

        # Save output to file using input file's base name
        input_base = os.path.splitext(os.path.basename(file_path))[0]
        outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        if fmt == 'srt':
            srt_filename = f"{input_base}.srt"
            srt_path = os.path.join(outputs_dir, srt_filename)
            # Generate SRT from segments and apply replacements
            def format_timestamp(seconds):
                ms = int((seconds - int(seconds)) * 1000)
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            srt_lines = []
            for idx, seg in enumerate(result.get("segments", []), 1):
                seg_text = seg["text"]
                for src, tgt in replacements:
                    seg_text = re.sub(re.escape(src), tgt, seg_text, flags=re.IGNORECASE)
                srt_lines.append(str(idx))
                srt_lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
                srt_lines.append(seg_text)
                srt_lines.append("")
            srt_content = "\n".join(srt_lines)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            output_text = srt_content
            output_file_path = srt_path
        else:
            # Save TXT output for download
            txt_filename = f"{input_base}.txt"
            txt_path = os.path.join(outputs_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            output_file_path = txt_path

        print(f"[DEBUG] Job {job_id} completed successfully.")
        transcription_jobs[job_id].update({'state': 'SUCCESS', 'progress': 100, 'output': output_text, 'output_file': output_file_path})
    except Exception as e:
        print(f"[ERROR] Exception in job {job_id}: {e}")
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        transcription_jobs[job_id].update({'state': 'FAILURE', 'progress': 100, 'error': error_msg})
    except Exception as e:
        transcription_jobs[job_id].update({'state': 'FAILURE', 'progress': 100, 'error': str(e)})

def start_transcription(file_path, output_dir, model, fmt, cpu, translate_zh):
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=transcribe_task, args=(job_id, file_path, output_dir, model, fmt, cpu, translate_zh))
    thread.start()
    return job_id

def get_job_status(job_id):
    return transcription_jobs.get(job_id, None)
