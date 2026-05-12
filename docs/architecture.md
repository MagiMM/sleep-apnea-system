# Architecture

## Target flow
1. Desktop app records microphone audio locally.
2. Desktop splits audio into 10-second windows.
3. Desktop sends raw WAV windows to the backend API.
4. Backend preprocesses audio.
5. Backend extracts the same features used in training.
6. Backend normalizes the features.
7. Backend runs inference and returns apnea / no_apnea with confidence.

## Guiding rule
Keep all feature extraction logic in the Python backend so training and inference stay aligned.
