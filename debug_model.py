#!/usr/bin/env python3
"""Diagnostic script to test model predictions on different audio inputs."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import librosa
import joblib

# Setup paths
ml_dir = Path("ml/artifacts")
model_path = ml_dir / "model_apnea.keras"
scaler_path = ml_dir / "scaler.joblib"

if not model_path.exists():
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)

print(f"[INFO] Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

print(f"[INFO] Loading scaler from {scaler_path}...")
scaler = joblib.load(scaler_path)

print(f"[OK] Model loaded. Input shape: {model.input_shape}")
print(f"[OK] Scaler loaded. Expected features: {scaler.n_features_in_}")

# Settings
SR = 16000
WINDOW_S = 10
MAX_SAMPLES = SR * WINDOW_S

def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract 160 features from audio."""
    features: list[float] = []
    
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
        features.extend(np.nanmean(mfcc, axis=1).tolist())
        features.extend(np.nanstd(mfcc, axis=1).tolist())
    except Exception as e:
        print(f"  [WARN] MFCC error: {e}")
        features.extend([0.0] * 26)
    
    try:
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.nanmean(mel_db, axis=1).tolist())
        features.extend(np.nanstd(mel_db, axis=1).tolist())
    except Exception as e:
        print(f"  [WARN] Mel spectrogram error: {e}")
        features.extend([0.0] * 128)
    
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([float(np.nanmean(zcr)), float(np.nanstd(zcr))])
    except Exception as e:
        print(f"  [WARN] ZCR error: {e}")
        features.extend([0.0, 0.0])
    
    try:
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=SR)
        features.extend([float(np.nanmean(spec_cent)), float(np.nanstd(spec_cent))])
    except Exception as e:
        print(f"  [WARN] Spectral centroid error: {e}")
        features.extend([0.0, 0.0])
    
    try:
        rms = librosa.feature.rms(y=audio)
        features.extend([float(np.nanmean(rms)), float(np.nanstd(rms))])
    except Exception as e:
        print(f"  [WARN] RMS error: {e}")
        features.extend([0.0, 0.0])
    
    vector = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

def predict(audio: np.ndarray, label: str) -> None:
    """Predict and print results."""
    print(f"\n[TEST] Testing: {label}")
    print(f"   Audio shape: {audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.6f}")
    
    # Pad/truncate to 10 seconds
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    elif len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")
    
    # Extract features
    features = extract_features(audio)
    print(f"   Features shape: {features.shape}")
    print(f"   Features stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.6f}")
    print(f"   First 5 features: {features[:5]}")
    
    # Normalize with scaler
    features_scaled = scaler.transform(features.reshape(1, -1))
    print(f"   Scaled stats: min={features_scaled.min():.4f}, max={features_scaled.max():.4f}, mean={features_scaled.mean():.6f}")
    print(f"   First 5 scaled: {features_scaled[0, :5]}")
    
    # Predict
    proba = float(model.predict(features_scaled, verbose=0).flatten()[0])
    label_pred = "apnea" if proba >= 0.5 else "no_apnea"
    print(f"   [OK] Prediction: {label_pred} (probability: {proba:.4f})")

# Test cases
print("\n" + "="*70)
print("TEST 1: SILENCE (all zeros)")
print("="*70)
silence = np.zeros(MAX_SAMPLES, dtype=np.float32)
predict(silence, "Silence")

print("\n" + "="*70)
print("TEST 2: WHITE NOISE")
print("="*70)
white_noise = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.1
predict(white_noise, "White noise")

print("\n" + "="*70)
print("TEST 3: DIFFERENT WHITE NOISE")
print("="*70)
white_noise2 = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.2
predict(white_noise2, "White noise (higher amplitude)")

print("\n" + "="*70)
print("TEST 4: SINE WAVE 440 Hz")
print("="*70)
t = np.arange(MAX_SAMPLES) / SR
sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
predict(sine_wave, "Sine wave 440 Hz")

print("\n" + "="*70)
print("TEST 5: SINE WAVE 880 Hz")
print("="*70)
sine_wave2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
predict(sine_wave2, "Sine wave 880 Hz")

print("\n" + "="*70)
print("TEST 6: NOISE BURST")
print("="*70)
noise_burst = np.zeros(MAX_SAMPLES, dtype=np.float32)
noise_burst[20000:40000] = np.random.randn(20000) * 0.3
predict(noise_burst, "Noise burst in middle")

print("\n" + "="*70)
print("[OK] Diagnostic complete!")
print("="*70)
