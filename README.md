# Sleep Apnea System

Monorepo for the sleep apnea detection product.

## Structure
- `apps/backend` - FastAPI inference service
- `apps/desktop` - desktop client for audio capture and 10 s windowing
- `ml/research` - notebook-based model research and experimentation
- `ml/artifacts` - exported model files, scaler, and feature config
- `shared` - shared schemas and contract notes
- `infra` - Docker and reverse proxy configuration
- `docs` - architecture and implementation notes

## First step
Start with the backend service, because it defines the model contract and the inference pipeline.

## Run backend locally
1. Open terminal in `apps/backend`.
2. Install dependencies: `uv sync`
3. Start API: `uv run uvicorn backend.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload`

## Run with Docker Compose
1. Open terminal in `infra`.
2. Run: `docker compose up --build`
3. API will be available through Nginx on `http://localhost:8080`.

## Current API
- `GET /health`
- `POST /predict-window` with `multipart/form-data` and `file` field containing a raw 10-second audio window.
