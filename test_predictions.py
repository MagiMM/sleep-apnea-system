#!/usr/bin/env python3
"""Test backend predictions with different audio inputs."""

import numpy as np
import tempfile
import wave
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
SR = 16000
WINDOW_S = 10
MAX_SAMPLES = SR * WINDOW_S

def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save audio to WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())

def test_audio(audio: np.ndarray, label: str) -> None:
    """Test audio and print predictions."""
    print(f"\n[TEST] Testing: {label}")
    print(f"   Audio: min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.6f}")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    
    save_wav(temp_path, audio, SR)
    
    try:
        # Send to API
        with temp_path.open("rb") as f:
            files = {"file": (temp_path.name, f, "application/octet-stream")}
            response = requests.post(f"{API_URL}/predict-window", files=files, timeout=10)
        
        if response.ok:
            data = response.json()
            prob = data.get("apnea_probability", -1)
            pred = data.get("label", "?")
            conf = data.get("confidence", -1)
            print(f"   [OK] {pred.upper()} | probability={prob:.4f} | confidence={conf:.4f}")
        else:
            print(f"   [ERROR] API error: {response.status_code}")
    finally:
        temp_path.unlink(missing_ok=True)

# Test cases
print("="*70)
print("BACKEND API PREDICTION TEST")
print("="*70)

print("\n1) SILENCE (all zeros)")
silence = np.zeros(MAX_SAMPLES, dtype=np.float32)
test_audio(silence, "Silence")

print("\n2) WHITE NOISE (low amplitude)")
white_noise = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.05
test_audio(white_noise, "White noise (0.05)")

print("\n3) WHITE NOISE (different random seed - low)")
np.random.seed(42)
white_noise2 = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.05
test_audio(white_noise2, "White noise (seed=42, 0.05)")

print("\n4) WHITE NOISE (higher amplitude)")
white_noise3 = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.3
test_audio(white_noise3, "White noise (0.3)")

print("\n5) SINE WAVE 440 Hz")
t = np.arange(MAX_SAMPLES) / SR
sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
test_audio(sine_wave, "Sine wave 440 Hz")

print("\n6) SINE WAVE 880 Hz")
sine_wave2 = np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.3
test_audio(sine_wave2, "Sine wave 880 Hz")

print("\n7) PINK NOISE")
# Simple pink noise approximation
import scipy.signal as signal
pink = signal.firwin(1000, 0.1) 
white = np.random.randn(MAX_SAMPLES) * 0.1
pink_noise = np.convolve(white, pink, mode='same').astype(np.float32)
pink_noise = (pink_noise / np.max(np.abs(pink_noise)) * 0.2).astype(np.float32)
test_audio(pink_noise, "Pink noise")

print("\n8) VERY LOUD NOISE")
loud_noise = np.random.randn(MAX_SAMPLES).astype(np.float32) * 0.8
test_audio(loud_noise, "Loud noise (0.8)")

print("\n" + "="*70)
print("[OK] Test complete!")
print("="*70)
