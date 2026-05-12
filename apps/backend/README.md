# Backend Service

FastAPI service for sleep apnea window inference.

## Local run (uv)
1. Change directory to `apps/backend`.
2. Install dependencies:
   - `uv sync`
3. Start API:
   - `uv run uvicorn backend.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload`

## Endpoints
- `GET /health`
- `POST /predict-window` (multipart/form-data, file field: `file`)

## Model artifacts
The service reads artifacts from `ml/artifacts` in repo root:
- `model_apnea.keras` (required)
- `scaler.joblib` (optional but recommended)

## Example request
`curl -X POST "http://localhost:8000/predict-window" -F "file=@window.wav"`
