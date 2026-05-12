from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    scaler_loaded: bool


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    apnea_probability: float
    sample_rate: int
    window_seconds: int
