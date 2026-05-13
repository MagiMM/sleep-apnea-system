# Sleep Apnea Detection System

Monorepo for the sleep apnea detection application with real-time microphone recording, automatic 10-second windowing, and live visualization.

## Structure
- `apps/backend` - FastAPI inference service for audio prediction
- `apps/desktop` - Tkinter desktop GUI for recording and analysis
- `ml/research` - Jupyter notebooks for model research and experimentation
- `ml/artifacts` - Exported model files (model_apnea.keras, scaler.joblib, feature_config.json)
- `docs` - Architecture and technical documentation
- `infra` - Docker and deployment configuration

## Prerequisites

- **Python 3.10+** (desktop) or **3.12+** (backend)
- **uv** package manager ([install](https://docs.astral.sh/uv/))
- **macOS/Windows/Linux** (tested on macOS)
- Microphone for audio recording

## Quick Start

### 1. Setup Backend (Terminal 1)

```bash
cd apps/backend
uv sync
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Verify with:
```bash
curl http://127.0.0.1:8000/health
```

### 2. Setup Desktop (Terminal 2)

```bash
cd apps/desktop
uv sync
python3 app.py
```

The GUI will open. Enter backend URL (default: `http://127.0.0.1:8000`) and click **Health** to verify connection.

## Features

### Backend API
- **GET /health** - Health check returning pipeline status
- **POST /predict-window** - Predict from audio file (multipart/form-data)
- **POST /predict-features** - Predict from pre-extracted features (160D numpy array)
- **POST /predict-signal-npy** - Predict from raw audio samples (numpy array)

### Desktop Application
- Real-time microphone recording with automatic **10-second windowing**
- Live predictions sent to backend during recording
- History tracking with timestamp, endpoint, label, and probability
- Interactive chart showing predictions over time (red = apnea, green = normal, blue = silence)
- Manual file upload for offline analysis

## How It Works

1. **Recording**: Click "Start Recording" to capture audio from microphone
2. **Windowing**: Audio is automatically split into 10-second chunks (160,000 samples @ 16kHz)
3. **Prediction**: Each chunk is sent to `/predict-window` endpoint
4. **Visualization**: Results appear in history table and live chart
5. **Stopping**: Click "Stop Recording" to end capture

## Audio Specifications
- Sample rate: 16,000 Hz
- Window size: 10 seconds (160,000 samples)
- Format: PCM WAV files
- Silence detection: max amplitude < 0.001 filtered as non-apnea

## Run with Docker Compose

```bash
cd infra
docker compose up --build
```

API will be available through Nginx at `http://localhost:8080`.

## Documentation

See [analiza-technologiczna.md](docs/analiza-technologiczna.md) for detailed technical stack analysis, architecture diagrams, and operational commands.
