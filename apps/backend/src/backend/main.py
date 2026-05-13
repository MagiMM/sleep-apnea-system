from io import BytesIO

import librosa
from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import ApneaInferencePipeline
from .schemas import HealthResponse, PredictionResponse
from .settings import settings
import numpy as np

app = FastAPI(title="Sleep Apnea API", version="0.1.0")
pipeline: ApneaInferencePipeline | None = None


@app.on_event("startup")
def load_pipeline() -> None:
    global pipeline
    pipeline = ApneaInferencePipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = pipeline is not None
    scaler_loaded = bool(ready and pipeline.scaler is not None)
    return HealthResponse(status="ok" if ready else "starting", pipeline_ready=ready, scaler_loaded=scaler_loaded)


@app.post("/predict-window", response_model=PredictionResponse)
async def predict_window(file: UploadFile = File(...)) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        audio, _ = librosa.load(BytesIO(audio_bytes), sr=settings.sample_rate, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {exc}") from exc

    result = pipeline.predict(audio)
    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        apnea_probability=result.apnea_probability,
        sample_rate=settings.sample_rate,
        window_seconds=settings.window_seconds,
    )


    @app.post("/predict-features", response_model=PredictionResponse)
    async def predict_features(file: UploadFile = File(...)) -> PredictionResponse:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Inference pipeline is not initialized")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            features = np.load(BytesIO(file_bytes))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load .npy file: {exc}") from exc

        if not isinstance(features, np.ndarray):
            raise HTTPException(status_code=400, detail="File must contain a numpy array")

        try:
            result = pipeline.predict_from_features(features)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error during prediction: {exc}") from exc

        return PredictionResponse(
            label=result.label,
            confidence=result.confidence,
            apnea_probability=result.apnea_probability,
            sample_rate=settings.sample_rate,
            window_seconds=settings.window_seconds,
        )
