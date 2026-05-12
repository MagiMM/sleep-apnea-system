from __future__ import annotations

import os
from pathlib import Path

import kagglehub
import librosa
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 64
MAX_SAMPLES = 160000
SEED = 42


def extract_features_safe(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    features: list[float] = []

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.nanmean(mfcc, axis=1).tolist())
        features.extend(np.nanstd(mfcc, axis=1).tolist())
    except Exception:
        features.extend([0.0] * (N_MFCC * 2))

    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.nanmean(mel_spec_db, axis=1).tolist())
        features.extend(np.nanstd(mel_spec_db, axis=1).tolist())
    except Exception:
        features.extend([0.0] * (N_MELS * 2))

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


def process_file(filepath: str, label: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        data = np.load(filepath)

        if data.ndim != 2:
            return None, None

        all_features = []
        for i in range(data.shape[0]):
            audio = data[i, :MAX_SAMPLES]
            if len(audio) < MAX_SAMPLES:
                audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")
            all_features.append(extract_features_safe(audio))

        return np.array(all_features), np.full(len(all_features), label)
    except Exception as exc:
        print(f"Failed processing {filepath}: {exc}")
        return None, None


def main() -> None:
    np.random.seed(SEED)

    dataset_path = kagglehub.dataset_download("bryandarquea/psg-audio-apnea-audios")
    psg_path = os.path.join(dataset_path, "PSG-AUDIO")

    apnea_files: list[str] = []
    no_apnea_files: list[str] = []

    for root, _, files in os.walk(psg_path):
        for file_name in files:
            if file_name.endswith("_ap.npy"):
                apnea_files.append(os.path.join(root, file_name))
            elif file_name.endswith("_nap.npy"):
                no_apnea_files.append(os.path.join(root, file_name))

    x_data: list[np.ndarray] = []

    print("Processing apnea files")
    for filepath in tqdm(apnea_files):
        features, _ = process_file(filepath, label=1)
        if features is not None:
            x_data.append(features)

    print("Processing non-apnea files")
    for filepath in tqdm(no_apnea_files):
        features, _ = process_file(filepath, label=0)
        if features is not None:
            x_data.append(features)

    if not x_data:
        raise RuntimeError("No features extracted; scaler cannot be fitted")

    x = np.vstack(x_data)
    scaler = StandardScaler()
    scaler.fit(x)

    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "scaler.joblib"
    joblib.dump(scaler, output_path)

    print(f"Saved scaler to {output_path}")
    print(f"Feature shape: {x.shape}")


if __name__ == "__main__":
    main()
