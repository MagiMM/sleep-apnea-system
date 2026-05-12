from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path = Path(__file__).resolve().parents[4]
    model_dir: Path = repo_root / "ml" / "artifacts"
    model_file: str = "model_apnea.keras"
    scaler_file: str = "scaler.joblib"
    sample_rate: int = 16000
    window_seconds: int = 10
    n_mfcc: int = 13
    n_mels: int = 64
    decision_threshold: float = 0.5


settings = Settings()
