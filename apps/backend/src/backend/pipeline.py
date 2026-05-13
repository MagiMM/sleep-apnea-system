from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import tensorflow as tf

from .settings import settings


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float
    apnea_probability: float


def _safe_load_scaler(scaler_path: Path) -> Any | None:
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)


class ApneaInferencePipeline:
    def __init__(self) -> None:
        model_path = settings.model_dir / settings.model_file
        scaler_path = settings.model_dir / settings.scaler_file

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = _safe_load_scaler(scaler_path)
        self.max_samples = settings.sample_rate * settings.window_seconds

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if audio.ndim != 1:
            audio = np.ravel(audio)

        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode="constant")

        return audio

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        features: list[float] = []
        sr = settings.sample_rate

        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=settings.n_mfcc)
            features.extend(np.nanmean(mfcc, axis=1).tolist())
            features.extend(np.nanstd(mfcc, axis=1).tolist())
        except Exception:
            features.extend([0.0] * (settings.n_mfcc * 2))

        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=settings.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.extend(np.nanmean(mel_db, axis=1).tolist())
            features.extend(np.nanstd(mel_db, axis=1).tolist())
        except Exception:
            features.extend([0.0] * (settings.n_mels * 2))

        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([float(np.nanmean(zcr)), float(np.nanstd(zcr))])
        except Exception:
            features.extend([0.0, 0.0])

        try:
            spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.extend([float(np.nanmean(spec_cent)), float(np.nanstd(spec_cent))])
        except Exception:
            features.extend([0.0, 0.0])

        try:
            rms = librosa.feature.rms(y=audio)
            features.extend([float(np.nanmean(rms)), float(np.nanstd(rms))])
        except Exception:
            features.extend([0.0, 0.0])

        vector = np.asarray(features, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, audio: np.ndarray) -> InferenceResult:
        processed_audio = self.preprocess_audio(audio)
        feature_vector = self.extract_features(processed_audio).reshape(1, -1)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        proba = float(self.model.predict(feature_vector, verbose=0).flatten()[0])
        apnea = proba >= settings.decision_threshold
        label = "apnea" if apnea else "no_apnea"
        confidence = proba if apnea else 1.0 - proba

        return InferenceResult(label=label, confidence=confidence, apnea_probability=proba)

        def predict_from_features(self, feature_vector: np.ndarray) -> InferenceResult:
            """Predict from pre-extracted features (e.g., from .npy file)."""
            # Ensure correct shape
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)
        
            feature_vector = np.asarray(feature_vector, dtype=np.float32)
        
            # Apply scaler if available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
        
            # Make prediction
            proba = float(self.model.predict(feature_vector, verbose=0).flatten()[0])
            apnea = proba >= settings.decision_threshold
            label = "apnea" if apnea else "no_apnea"
            confidence = proba if apnea else 1.0 - proba

            return InferenceResult(label=label, confidence=confidence, apnea_probability=proba)
