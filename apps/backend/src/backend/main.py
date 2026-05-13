from io import BytesIO

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import ApneaInferencePipeline
from .schemas import HealthResponse, PredictionResponse
from .settings import settings

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

    file_name = file.filename or "uploaded file"
    if file_name.endswith("_types.npy"):
        raise HTTPException(
            status_code=400,
            detail=(
                "You uploaded a *_types.npy file (labels/event types). "
                "Upload extracted feature vectors (.npy with 160 features per sample) "
                "or use /predict-window for audio files."
            ),
        )

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


@app.post("/predict-signal-npy", response_model=PredictionResponse)
async def predict_signal_npy(file: UploadFile = File(...)) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized")

    file_name = file.filename or "uploaded file"
    if file_name.endswith("_types.npy"):
        raise HTTPException(
            status_code=400,
            detail="*_types.npy contains labels, not signal windows. Upload *_ap.npy or *_nap.npy.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        signals = np.load(BytesIO(file_bytes), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to load .npy file: {exc}") from exc

    if not isinstance(signals, np.ndarray):
        raise HTTPException(status_code=400, detail="File must contain a numpy array")

    if signals.ndim == 1:
        result = pipeline.predict(signals)
        return PredictionResponse(
            label=result.label,
            confidence=result.confidence,
            apnea_probability=result.apnea_probability,
            sample_rate=settings.sample_rate,
            window_seconds=settings.window_seconds,
        )

    if signals.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected 1D or 2D numpy array with raw signal samples")

    if signals.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Uploaded array has no windows")

    probabilities: list[float] = []
    for i in range(signals.shape[0]):
        window_result = pipeline.predict(signals[i])
        probabilities.append(window_result.apnea_probability)

    apnea_probability = float(np.mean(probabilities))
    label = "apnea" if apnea_probability >= settings.decision_threshold else "no_apnea"
    confidence = apnea_probability if label == "apnea" else 1.0 - apnea_probability

    return PredictionResponse(
        label=label,
        confidence=confidence,
        apnea_probability=apnea_probability,
        sample_rate=settings.sample_rate,
        window_seconds=settings.window_seconds,
    )
