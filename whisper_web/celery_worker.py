
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
        'stage': 'transcribing',
        'transcribe_progress': 0,
        'translate_progress': 0,
        'post_progress': 0,
        'start_time': start_time
    }
    import traceback
    try:
        import whisper
        import torch
        print("[CUDA] celery_worker.py: CUDA available:", torch.cuda.is_available())
        import warnings
        import re
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        transcription_jobs[job_id].update({'state': 'PROGRESS', 'progress': 5})
        model_name = model or 'base'
        print(f"[DEBUG] Loading Whisper model: {model_name} on device: {device}")
        model_obj = whisper.load_model(model_name, device=device)
        print(f"[DEBUG] Model loaded successfully.")

        # --- No chunking: transcribe the whole audio file at once ---
        print(f"[DEBUG] No chunking, transcribing the whole audio file...")
        transcription_jobs[job_id].update({'stage': 'transcribing', 'transcribe_progress': 0})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp16 = device in ["mps", "cuda"]
            result = model_obj.transcribe(file_path, verbose=False, fp16=fp16)
        transcription_jobs[job_id].update({'transcribe_progress': 100, 'progress': 50})
        if translate_zh:
            transcription_jobs[job_id].update({'stage': 'translating', 'translate_progress': 0})
            print(f"[DEBUG] Starting local MarianMT + OpenCC translation to Traditional Chinese...")
            try:
                from transformers import MarianMTModel, MarianTokenizer
                from opencc import OpenCC
                cc = OpenCC('s2t')  # 簡體轉繁體
                model_dir = os.path.join(os.path.dirname(__file__), '../translation_model')
                os.makedirs(model_dir, exist_ok=True)
                model_name = "Helsinki-NLP/opus-mt-en-zh"
                tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=model_dir)
                model = MarianMTModel.from_pretrained(model_name, cache_dir=model_dir)
                # 分批翻譯主文本
                sentences = [s.strip() for s in re.split(r'(?<=[.!?。！？])\s+', result["text"]) if s.strip()]
                batch_size = 8
                zh_sents_trad = []
                for i in range(0, len(sentences), batch_size):
                    batch_sents = sentences[i:i+batch_size]
                    batch = tokenizer(batch_sents, return_tensors="pt", padding=True, truncation=True)
                    translated = model.generate(**batch)
                    zh_sents = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                    zh_sents_trad.extend([cc.convert(s) for s in zh_sents])
                    # 進度回報
                    percent = int(100 * (i + batch_size) / max(len(sentences), 1))
                    transcription_jobs[job_id].update({'translate_progress': min(percent, 99)})
                zh_text = ' '.join(zh_sents_trad)
                result["text"] = zh_text
                print(f"[DEBUG] Main text batch translated (MarianMT+OpenCC)。")
                # 分批翻譯 segments
                segments = result.get("segments", [])
                seg_texts = [seg["text"] for seg in segments]
                zh_seg_sents_trad = []
                if seg_texts:
                    for i in range(0, len(seg_texts), batch_size):
                        batch_sents = seg_texts[i:i+batch_size]
                        seg_batch = tokenizer(batch_sents, return_tensors="pt", padding=True, truncation=True)
                        seg_translated = model.generate(**seg_batch)
                        zh_seg_sents = [tokenizer.decode(t, skip_special_tokens=True) for t in seg_translated]
                        zh_seg_sents_trad.extend([cc.convert(s) for s in zh_seg_sents])
                        # 進度回報
                        percent = int(100 * (i + batch_size) / max(len(seg_texts), 1))
                        transcription_jobs[job_id].update({'translate_progress': min(percent, 99)})
                    # 對齊 segment 數量
                    for i, seg in enumerate(segments):
                        if i < len(zh_seg_sents_trad):
                            seg["text"] = zh_seg_sents_trad[i]
                        else:
                            seg["text"] = ""
                print(f"[DEBUG] Segments batch translated (MarianMT+OpenCC)。")
                # Debug: 印出每個 segment 翻譯前後內容
                for i, seg in enumerate(segments):
                    orig = seg_texts[i] if i < len(seg_texts) else ""
                    print(f"[DEBUG] seg[{i}] before translation: {orig}")
                    print(f"[DEBUG] seg[{i}] after translation:  {seg['text']}")
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
        # Apply replacements to output_text，忽略所有空白（半形、全形）
        def make_space_insensitive_pattern(src):
            # 將 src 轉為 pattern，忽略所有空白（半形、全形）
            import re
            chars = [c for c in src if not c.isspace()]
            # [\s\u3000]* 代表可有可無的半形或全形空白
            return r''.join([re.escape(c) + r'[\s\u3000]*' for c in chars])

        for src, tgt in replacements:
            pattern = make_space_insensitive_pattern(src)
            output_text = re.sub(pattern, tgt, output_text, flags=re.IGNORECASE)

        # Save output to file using input file's base name
        # 在輸出前，自動合併所有中文之間多餘的空白
        import re as _re
        def merge_chinese_spaces(text):
            # 合併所有「中文+空白+中文」為「中文中文」
            return _re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)

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
            # 先依 start 時間排序，避免 SRT 時間錯亂
            segments = sorted(result.get("segments", []), key=lambda seg: seg["start"])
            srt_lines = []
            import unicodedata
            def normalize_text(text):
                # 移除所有空白、全形空白，並轉半形
                text = ''.join(text.split())
                text = unicodedata.normalize('NFKC', text)
                return text
            for idx, seg in enumerate(segments, 1):
                seg_text = seg["text"]
                orig_text = seg_text
                for src, tgt in replacements:
                    pattern = make_space_insensitive_pattern(src)
                    seg_text_new = re.sub(pattern, tgt, seg_text, flags=re.IGNORECASE)
                    if seg_text_new != seg_text:
                        seg_text = seg_text_new
                # 合併中文間多餘空白
                seg_text = merge_chinese_spaces(seg_text)
                if orig_text != seg_text:
                    print(f"[DEBUG] SRT seg[{idx}] before dict: {orig_text}")
                    print(f"[DEBUG] SRT seg[{idx}] after dict:  {seg_text}")
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
            # 合併中文間多餘空白
            output_text = merge_chinese_spaces(output_text)
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
