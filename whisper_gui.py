import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os

# Path to your transcribe.py script
TRANSCRIBE_SCRIPT = os.path.join(os.path.dirname(__file__), 'transcribe.py')

class WhisperGUI:
    def __init__(self):
        # Try to use TkinterDnD for drag-and-drop
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            self.root = TkinterDnD.Tk()
            self.dnd_enabled = True
            self.DND_FILES = DND_FILES
        except ImportError:
            self.root = tk.Tk()
            self.dnd_enabled = False
        self.root.title("Whisper Transcription GUI")
        self.root.geometry("500x350")
        self.file_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model = tk.StringVar(value="base")
        self.format = tk.StringVar(value="srt")
        self.cpu = tk.BooleanVar()
        self.translate_zh = tk.BooleanVar()
        self.create_widgets()

    def create_widgets(self):
        # File selection
        file_frame = ttk.LabelFrame(self.root, text="Audio/Video File")
        file_frame.pack(fill="x", padx=10, pady=5)
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=40)
        file_entry.pack(side="left", padx=5, pady=5)
        file_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        file_btn.pack(side="left", padx=5)
        drag_label = ttk.Label(file_frame, text="(Or drag file here)")
        drag_label.pack(side="left", padx=5)
        # Drag-and-drop support using tkinterdnd2
        if self.dnd_enabled:
            file_entry.drop_target_register(self.DND_FILES)
            file_entry.dnd_bind('<<Drop>>', self.drop_file)
        else:
            print("[WARNING] tkinterdnd2 not installed, drag-and-drop disabled.")

        # Output directory
        out_frame = ttk.LabelFrame(self.root, text="Output Directory (optional)")
        out_frame.pack(fill="x", padx=10, pady=5)
        out_entry = ttk.Entry(out_frame, textvariable=self.output_dir, width=40)
        out_entry.pack(side="left", padx=5, pady=5)
        out_btn = ttk.Button(out_frame, text="Browse", command=self.browse_output)
        out_btn.pack(side="left", padx=5)

        # Model selection
        model_frame = ttk.LabelFrame(self.root, text="Model")
        model_frame.pack(fill="x", padx=10, pady=5)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model, values=["base", "large-v3-turbo", "large-v3", "small", "tiny"])
        model_combo.pack(fill="x", padx=5, pady=5)

        # Format selection
        format_frame = ttk.LabelFrame(self.root, text="Output Format")
        format_frame.pack(fill="x", padx=10, pady=5)
        format_combo = ttk.Combobox(format_frame, textvariable=self.format, values=["txt", "json", "srt", "all"])
        format_combo.pack(fill="x", padx=5, pady=5)

        # Options
        options_frame = ttk.LabelFrame(self.root, text="Options")
        options_frame.pack(fill="x", padx=10, pady=5)
        cpu_check = ttk.Checkbutton(options_frame, text="Force CPU", variable=self.cpu)
        cpu_check.pack(side="left", padx=5)
        zh_check = ttk.Checkbutton(options_frame, text="Translate to Traditional Chinese", variable=self.translate_zh)
        zh_check.pack(side="left", padx=5)

        # Transcribe button
        transcribe_btn = ttk.Button(self.root, text="Transcribe", command=self.run_transcribe)
        transcribe_btn.pack(pady=15)

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Audio/Video File")
        if file_path:
            self.file_path.set(file_path)

    def browse_output(self):
        out_dir = filedialog.askdirectory(title="Select Output Directory")
        if out_dir:
            self.output_dir.set(out_dir)

    def drop_file(self, event):
        # Set the file path from drag-and-drop event
        if event.data:
            # Remove curly braces if present (TkinterDnD wraps paths in {})
            file_path = event.data.strip('{}')
            self.file_path.set(file_path)

    # No longer needed: drop_target_register

    def run_transcribe(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an audio/video file.")
            return
        cmd = ["python3", TRANSCRIBE_SCRIPT, file_path]
        if self.output_dir.get():
            cmd += ["-o", self.output_dir.get()]
        if self.model.get():
            cmd += ["-m", self.model.get()]
        if self.format.get():
            cmd += ["-f", self.format.get()]
        if self.cpu.get():
            cmd += ["--cpu"]
        if self.translate_zh.get():
            cmd += ["--translate-zh"]
        self.run_command(cmd)

    def run_command(self, cmd):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                messagebox.showinfo("Success", "Transcription completed!\n" + result.stdout)
            else:
                messagebox.showerror("Error", "Transcription failed!\n" + result.stderr)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run transcription: {e}")

if __name__ == "__main__":
    app = WhisperGUI()
    app.root.mainloop()
