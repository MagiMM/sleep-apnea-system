from __future__ import annotations

import json
import importlib
import tempfile
import threading
import time
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import requests


class DesktopClientApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sleep Apnea Desktop Client")
        self.geometry("980x700")
        self.minsize(860, 620)

        self.backend_url = tk.StringVar(value="http://127.0.0.1:8000")
        self.file_path = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Ready")
        self.sample_rate = 16000
        self.history: list[dict[str, object]] = []
        self._recording = False
        self._record_thread: threading.Thread | None = None
        self._window_size = 10 * self.sample_rate

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.LabelFrame(root, text="Backend", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="API URL:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.backend_url).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Health", command=self._check_health).grid(row=0, column=2)
        top.columnconfigure(1, weight=1)

        file_box = ttk.LabelFrame(root, text="Input File", padding=10)
        file_box.pack(fill="x", pady=(10, 0))

        ttk.Entry(file_box, textvariable=self.file_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_box, text="Choose File", command=self._choose_file).grid(row=0, column=1, padx=(8, 0))
        self.record_button = ttk.Button(file_box, text="▶ Start Recording", command=self._toggle_recording)
        self.record_button.grid(row=0, column=2, padx=(16, 0))
        self.record_time_label = ttk.Label(file_box, text="", font=("TkDefaultFont", 10, "bold"))
        self.record_time_label.grid(row=0, column=3)
        file_box.columnconfigure(0, weight=1)

        actions = ttk.LabelFrame(root, text="Send To Endpoint", padding=10)
        actions.pack(fill="x", pady=(10, 0))

        ttk.Button(
            actions,
            text="Audio -> /predict-window",
            command=lambda: self._predict_with_file("/predict-window"),
        ).grid(row=0, column=0, sticky="ew")

        ttk.Button(
            actions,
            text="Features NPY -> /predict-features",
            command=lambda: self._predict_with_file("/predict-features"),
        ).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(
            actions,
            text="Signal NPY -> /predict-signal-npy",
            command=lambda: self._predict_with_file("/predict-signal-npy"),
        ).grid(row=0, column=2, sticky="ew")

        for idx in range(3):
            actions.columnconfigure(idx, weight=1)

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True, pady=(10, 0))

        output = ttk.LabelFrame(middle, text="Response", padding=10)
        output.pack(side="left", fill="both", expand=True)

        self.output_text = tk.Text(output, wrap="word", height=18)
        self.output_text.pack(fill="both", expand=True)

        side = ttk.Frame(middle)
        side.pack(side="left", fill="y", padx=(10, 0))

        chart_box = ttk.LabelFrame(side, text="Apnea Risk Trend", padding=10)
        chart_box.pack(fill="x")
        self.chart_canvas = tk.Canvas(chart_box, width=320, height=160, bg="#ffffff", highlightthickness=1, highlightbackground="#d0d0d0")
        self.chart_canvas.pack(fill="x")
        self.chart_canvas.bind("<Configure>", lambda e: self._schedule_draw_chart())

        history_box = ttk.LabelFrame(side, text="History", padding=10)
        history_box.pack(fill="both", expand=True, pady=(10, 0))

        self.history_tree = ttk.Treeview(
            history_box,
            columns=("time", "endpoint", "label", "prob", "status"),
            show="headings",
            height=14,
        )
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("endpoint", text="Endpoint")
        self.history_tree.heading("label", text="Label")
        self.history_tree.heading("prob", text="Apnea p")
        self.history_tree.heading("status", text="HTTP")
        self.history_tree.column("time", width=62, anchor="center")
        self.history_tree.column("endpoint", width=120, anchor="w")
        self.history_tree.column("label", width=70, anchor="center")
        self.history_tree.column("prob", width=70, anchor="e")
        self.history_tree.column("status", width=55, anchor="center")
        self.history_tree.pack(fill="both", expand=True)

        ttk.Button(history_box, text="Clear History", command=self._clear_history).pack(fill="x", pady=(8, 0))

        status_bar = ttk.Label(root, textvariable=self.status_text, anchor="w")
        status_bar.pack(fill="x", pady=(8, 0))

    def _set_status(self, text: str) -> None:
        self._ui(lambda: self.status_text.set(text))

    def _choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose input file",
            filetypes=[
                ("Audio or NPY", "*.wav *.mp3 *.ogg *.flac *.m4a *.npy"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.file_path.set(selected)

    def _toggle_recording(self) -> None:
        if self._recording:
            self._recording = False
            self._set_status("Stopping recording...")
            if self._record_thread and self._record_thread.is_alive():
                self._record_thread.join(timeout=2.0)
            self._ui(lambda: self.record_button.config(text="▶ Start Recording"))
            self._set_status("Recording stopped")
        else:
            self._recording = True
            self._record_thread = threading.Thread(target=self._record_stream, daemon=True)
            self._record_thread.start()
            self._ui(lambda: self.record_button.config(text="⏹ Stop Recording"))

    def _record_stream(self) -> None:
        sd = importlib.import_module("sounddevice")
        start_time = time.time()
        buffer: list[np.ndarray] = []
        self._set_status("Recording...")

        while self._recording:
            elapsed = time.time() - start_time
            self._ui(lambda e=elapsed: self.record_time_label.config(text=f"{int(e):02d}s"))
            
            frames = int(self.sample_rate * 1.0)
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            mono = np.squeeze(audio)
            buffer.append(mono)

            total_frames = sum(len(chunk) for chunk in buffer)
            if total_frames >= self._window_size:
                accumulated = np.concatenate(buffer)
                window_audio = accumulated[: self._window_size]
                buffer = [accumulated[self._window_size:]] if len(accumulated) > self._window_size else []

                self._process_window(window_audio, int(elapsed))

        self._ui(lambda: self.record_time_label.config(text=""))

    def _process_window(self, audio: np.ndarray, timestamp: int) -> None:
        self._set_status(f"Processing window at {timestamp}s...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            self._save_wav(temp_path, audio, self.sample_rate)
            self._send_prediction_and_record(temp_path, timestamp)
            temp_path.unlink(missing_ok=True)
        except Exception as exc:
            self._set_status(f"Window processing failed: {exc}")

    def _send_prediction_and_record(self, path: Path, timestamp: int) -> None:
        try:
            url = f"{self.backend_url.get().rstrip('/')}/predict-window"
            with path.open("rb") as handle:
                files = {"file": (path.name, handle, "application/octet-stream")}
                response = requests.post(url, files=files, timeout=120)

            payload: object
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw": response.text}

            if response.ok and isinstance(payload, dict):
                label = str(payload.get("label", "-"))
                prob = float(payload.get("apnea_probability", 0))
                item = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "endpoint": f"/predict-window (@{timestamp}s)",
                    "label": label,
                    "probability": prob,
                    "status": response.status_code,
                    "timestamp": timestamp,
                }
                self.history.append(item)
                self._ui(self._refresh_history_view)
            self._set_status("Recording...")
        except Exception as exc:
            self._set_status(f"Prediction failed: {exc}")

    def _check_health(self) -> None:
        self._run_async(self._health_request)

    def _save_wav(self, path: Path, audio: np.ndarray, sr: int) -> None:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(pcm.tobytes())

    def _health_request(self) -> None:
        try:
            url = f"{self.backend_url.get().rstrip('/')}/health"
            response = requests.get(url, timeout=15)
            self._render_response(response, endpoint="/health")
            self._set_status("Health checked")
        except requests.RequestException as exc:
            self._set_status("Health failed")
            self._write_output({"error": str(exc)})

    def _predict_with_file(self, endpoint: str) -> None:
        path = Path(self.file_path.get().strip())
        if not path.exists() or not path.is_file():
            messagebox.showerror("Missing file", "Choose an existing file first.")
            return

        self._run_async(lambda: self._upload_request(endpoint, path))

    def _upload_request(self, endpoint: str, path: Path) -> None:
        self._set_status(f"Sending {path.name} to {endpoint} ...")
        try:
            url = f"{self.backend_url.get().rstrip('/')}{endpoint}"
            with path.open("rb") as handle:
                files = {"file": (path.name, handle, "application/octet-stream")}
                response = requests.post(url, files=files, timeout=120)
            self._render_response(response, endpoint=endpoint)
            self._set_status(f"Done: {endpoint}")
        except requests.RequestException as exc:
            self._set_status("Request failed")
            self._write_output({"error": str(exc)})

    def _render_response(self, response: requests.Response, endpoint: str) -> None:
        payload: object
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text}

        rendered = {
            "status_code": response.status_code,
            "ok": response.ok,
            "payload": payload,
        }
        self._write_output(rendered)
        self._append_history(response.status_code, endpoint, payload)

    def _write_output(self, content: object) -> None:
        text = json.dumps(content, indent=2, ensure_ascii=True)
        self._ui(lambda: self._replace_text(text))

    def _replace_text(self, text: str) -> None:
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)

    def _append_history(self, status_code: int, endpoint: str, payload: object) -> None:
        label = "-"
        probability = None
        if isinstance(payload, dict):
            label = str(payload.get("label", "-"))
            if isinstance(payload.get("apnea_probability"), (int, float)):
                probability = float(payload["apnea_probability"])

        item = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "endpoint": endpoint,
            "label": label,
            "probability": probability,
            "status": status_code,
        }
        self.history.append(item)
        self._ui(self._refresh_history_view)

    def _refresh_history_view(self) -> None:
        self.history_tree.delete(*self.history_tree.get_children())
        for item in self.history[-120:]:
            prob = item["probability"]
            prob_text = f"{prob:.3f}" if isinstance(prob, float) else "-"
            self.history_tree.insert(
                "",
                "end",
                values=(item["time"], item["endpoint"], item["label"], prob_text, item["status"]),
            )
        self._draw_chart()

    def _draw_chart(self) -> None:
        canvas = self.chart_canvas
        canvas.delete("all")
        width = int(canvas.winfo_width() or 320)
        height = int(canvas.winfo_height() or 160)
        pad = 20

        canvas.create_line(pad, height - pad, width - pad, height - pad, fill="#999999", width=2)
        canvas.create_line(pad, pad, pad, height - pad, fill="#999999", width=2)
        canvas.create_text(12, pad - 4, text="APNEA", fill="#c84d2f", anchor="w", font=("TkDefaultFont", 9, "bold"))
        canvas.create_text(12, height - pad + 8, text="NORMAL", fill="#2eb82e", anchor="w", font=("TkDefaultFont", 9, "bold"))

        # Filter out invalid predictions (silence_detected has probability=-1.0)
        valid_history = [x for x in self.history if x.get("label") not in ("silence_detected",)]
        
        series = [float(x["probability"]) for x in valid_history if isinstance(x.get("probability"), float) and x.get("probability") >= 0]
        labels = [str(x.get("label", "-")) for x in valid_history if isinstance(x.get("probability"), float) and x.get("probability") >= 0]

        if len(series) < 1:
            canvas.create_text(width // 2, height // 2, text="No valid predictions", fill="#999999", anchor="center")
            return

        view = series[-30:]
        view_labels = labels[-30:]
        span_x = max(1, len(view) - 1)

        for i, val in enumerate(view):
            x = pad + (i / span_x) * (width - 2 * pad)
            y = (height - pad) - (val * (height - 2 * pad))
            label = view_labels[i] if i < len(view_labels) else "-"
            color = "#c84d2f" if label == "apnea" else "#2eb82e"
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)

        if len(view) > 1:
            points = []
            for i, val in enumerate(view):
                x = pad + (i / span_x) * (width - 2 * pad)
                y = (height - pad) - (val * (height - 2 * pad))
                points.extend([x, y])
            canvas.create_line(*points, fill="#999999", width=1, smooth=True)

        last_label = view_labels[-1] if view_labels else "-"
        last_val = view[-1] if view else 0
        canvas.create_text(width - 8, pad + 4, text=f"{last_label.upper()} ({last_val:.3f})", fill="#333333", anchor="ne", font=("TkDefaultFont", 9, "bold"))

    def _schedule_draw_chart(self) -> None:
        self.after(100, self._draw_chart)

    def _clear_history(self) -> None:
        self.history.clear()
        self._refresh_history_view()

    def _ui(self, fn) -> None:
        self.after(0, fn)

    def _run_async(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()


if __name__ == "__main__":
    DesktopClientApp().mainloop()
