import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from celery_worker import start_transcription, get_job_status
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'whisper_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve output files for download
@app.route('/download')
def download():
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import subprocess
from celery_worker import start_transcription, get_job_status
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'whisper_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your transcribe.py script
TRANSCRIBE_SCRIPT = os.path.join(os.path.dirname(__file__), '../transcribe.py')

MODELS = ["base", "large-v3-turbo", "large-v3", "small", "tiny"]
FORMATS = ["txt", "json", "srt", "all"]

def run_transcribe(file_path, output_dir, model, fmt, cpu, translate_zh):
    # Deprecated: now handled by Celery
    pass

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    task_id = None
    if request.method == 'POST':
        file = request.files.get('file')
        output_dir = request.form.get('output_dir')
        model = request.form.get('model')
        fmt = request.form.get('format')
        cpu = request.form.get('cpu') == 'on'
        translate_zh = request.form.get('translate_zh') == 'on'
        print(f"[LOG] Received POST: file={file.filename if file else None}, output_dir={output_dir}, model={model}, format={fmt}, cpu={cpu}, translate_zh={translate_zh}")
        if not file or file.filename == '':
            error = "Please select an audio/video file."
            print(f"[ERROR] {error}")
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"[LOG] Saved file to {file_path}")
            # Start transcription job in a background thread
            job_id = start_transcription(file_path, output_dir, model, fmt, cpu, translate_zh)
            print(f"[LOG] Started transcription job: {job_id}")
            # Show progress page
            return redirect(url_for('progress', task_id=job_id))
    return render_template('index.html', models=MODELS, formats=FORMATS, result=result, error=error, running=False)

@app.route('/progress/<task_id>')
def progress(task_id):
    return render_template('progress.html', task_id=task_id)

@app.route('/task_status/<task_id>')
def task_status(task_id):
    try:
        job = get_job_status(task_id)
        print(f"[DEBUG] task_status called for job_id={task_id}, job={job}")
        response = {}
        if job:
            response['state'] = job.get('state', 'PENDING')
            response['progress'] = job.get('progress', 0)
            response['chunk_progress'] = job.get('chunk_progress', 0)
            response['transcribe_progress'] = job.get('transcribe_progress', 0)
            response['translate_progress'] = job.get('translate_progress', 0)
            response['post_progress'] = job.get('post_progress', 0)
            response['stage'] = job.get('stage', '')
            response['start_time'] = job.get('start_time', None)
            if job.get('state') == 'SUCCESS':
                response['output'] = job.get('output', '')
                if 'output_file' in job:
                    response['output_file'] = job['output_file']
            if job.get('state') == 'FAILURE':
                response['error'] = job.get('error', 'Unknown error')
        else:
            response['state'] = 'PENDING'
            response['progress'] = 0
            response['chunk_progress'] = 0
            response['transcribe_progress'] = 0
            response['translate_progress'] = 0
            response['post_progress'] = 0
        return jsonify(response)
    except Exception as e:
        return jsonify({'state': 'FAILURE', 'progress': 100, 'error': f'Internal error: {e}'})

if __name__ == '__main__':
    app.run(debug=True)

# Install the required packages
import whisper
model = whisper.load_model("base")
result = model.transcribe("your_test_file.mp3")
print(result["text"])# RUN pip install flask celery redis openai-whisper torch deep-translator werkzeug
