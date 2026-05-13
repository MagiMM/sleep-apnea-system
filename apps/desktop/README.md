# Desktop app

Simple cross-platform desktop client (macOS and Windows) for testing backend endpoints.

## Current MVP
- set backend URL
- run health check
- upload a file to:
  - `/predict-window` (audio)
  - `/predict-features` (.npy with extracted features)
  - `/predict-signal-npy` (.npy with raw signal windows)
- record microphone audio and send directly to `/predict-window`
- prediction history table
- apnea risk trend chart
- view JSON response directly in the app

## Run
1. Open terminal in this folder.
2. Install dependencies:
	- `uv sync`
3. Start app:
	- `uv run python app.py`

## Build desktop package
Run these commands in `apps/desktop`.

1. Install PyInstaller:
  - `uv add --dev pyinstaller`
2. Build single app bundle/exe:
  - `uv run pyinstaller --onefile --windowed --name SleepApneaDesktop app.py`

Build output:
- macOS: `dist/SleepApneaDesktop`
- Windows: `dist/SleepApneaDesktop.exe`

## Notes
- The backend must be running first (default: `http://127.0.0.1:8000`).
- For dataset files, use `/predict-signal-npy` for `*_ap.npy` or `*_nap.npy`.
- Do not upload `*_types.npy` to prediction endpoints (those are labels).
- On macOS, grant microphone access to the app/terminal on first recording.
