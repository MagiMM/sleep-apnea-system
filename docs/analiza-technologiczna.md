# Szkic stosu technologicznego

## 1. Cel
Dokument opisuje technologie uzyte w systemie wykrywania bezdechu sennego i role kazdego komponentu.

## 2. Architektura wysokiego poziomu
- Aplikacja desktop (Python) przechwytuje audio i wysyla dane do API.
- Backend (FastAPI) wykonuje preprocessing, ekstrakcje cech i inferencje modelu.
- Artefakty ML (model i scaler) sa ladowane lokalnie przez backend.

Przeplyw danych:
1. Mikrofon -> desktop.
2. Desktop -> podzial na okna 10 s.
3. Desktop -> POST /predict-window (WAV).
4. Backend -> preprocessing i feature extraction.
5. Backend -> normalizacja cech i predykcja.
6. Backend -> odpowiedz JSON z wynikiem.

## 3. Backend
### 3.1 Framework API
- FastAPI: framework webowy w Pythonie do budowy REST API. W projekcie udostepnia endpointy inferencyjne (/health, /predict-window, /predict-features, /predict-signal-npy), walidacje danych wejsciowych i serializacje odpowiedzi JSON.
- Uvicorn: serwer ASGI uruchamiajacy aplikacje FastAPI. W projekcie odpowiada za obsluge zadan HTTP lokalnie (host 127.0.0.1, port 8000).
- python-multipart: biblioteka do obslugi formularzy multipart/form-data. W projekcie umozliwia przyjmowanie plikow audio i plikow .npy przesylanych przez klienta desktopowego.

### 3.2 Warstwa ML
- TensorFlow/Keras: framework do modeli uczenia maszynowego. W projekcie laduje wytrenowany model z pliku model_apnea.keras i wykonuje predykcje prawdopodobienstwa bezdechu.
- librosa: biblioteka do analizy sygnalow audio. W projekcie dekoduje audio do stalej czestotliwosci probkowania (16 kHz) oraz wylicza cechy akustyczne (m.in. MFCC, mel-spectrogram, ZCR, centroid, RMS).
- scikit-learn + joblib: narzedzia do przetwarzania cech i serializacji obiektow. W projekcie StandardScaler jest zapisany do pliku i ladowany przez joblib, a nastepnie wykorzystywany do normalizacji cech przed inferencja.
- numpy: biblioteka obliczen numerycznych na tablicach. W projekcie sluzy do preprocessingu sygnalu (przycinanie/padding), agregacji cech i obliczen pomocniczych.

### 3.3 Endpointy
- GET /health
- POST /predict-window
- POST /predict-features
- POST /predict-signal-npy

### 3.4 Konfiguracja
- Plik: apps/backend/src/backend/settings.py
- Kluczowe parametry:
	- sample_rate = 16000
	- window_seconds = 10
	- decision_threshold = 0.5

Rola konfiguracji:
- Utrzymuje stale parametry pipeline w jednym miejscu.
- Zapewnia spojne wartosci miedzy endpointami i warstwa inferencji.

## 4. Aplikacja desktop
### 4.1 UI i logika aplikacji
- tkinter: standardowa biblioteka GUI w Pythonie. W projekcie tworzy interfejs uzytkownika: formularz konfiguracji API, przyciski akcji, tabele historii i wykres trendu.
- threading: modul do pracy wielowatkowej. W projekcie uruchamia nagrywanie i zapytania HTTP w tle, aby interfejs pozostawal responsywny podczas analizy audio.

### 4.2 Audio i komunikacja
- sounddevice: biblioteka do obslugi wejsc audio z poziomu Pythona. W projekcie odpowiada za przechwytywanie sygnalu z mikrofonu w czasie rzeczywistym, czyli pobieranie kolejnych fragmentow probek audio do dalszej analizy.
- wave: standardowy modul Pythona do zapisu i odczytu plikow WAV. W projekcie sluzy do tymczasowego zapisu zarejestrowanego okna audio (10 s) do formatu WAV, ktory jest nastepnie wysylany do backendu.
- requests: biblioteka HTTP klienta. W projekcie realizuje komunikacje desktop -> backend (wysylanie pliku audio do endpointu /predict-window oraz odbior odpowiedzi JSON z wynikiem predykcji).
- numpy: podstawowa biblioteka obliczen numerycznych. W projekcie odpowiada za operacje na probkach audio (buforowanie, laczenie fragmentow, przycinanie do dlugosci okna, konwersje i przygotowanie danych do wysylki).

### 4.3 Funkcje klienta
- Start/Stop nagrywania: sterowanie sesja mikrofonu z poziomu GUI.
- Automatyczny podzial strumienia na okna 10 s: buforowanie probek i wydzielanie kolejnych fragmentow sygnalu o stalej dlugosci wymaganej przez pipeline.
- Wysylka okien do /predict-window: przekazanie kazdego okna audio do backendu i odbior wyniku inferencji.
- Historia predykcji i wykres trendu: prezentacja kolejnych wynikow w czasie, co ulatwia obserwacje zmian ryzyka miedzy oknami.

## 5. Artefakty ML i dane
- ml/artifacts/model_apnea.keras: wytrenowany model sieci neuronowej wykorzystywany podczas inferencji.
- ml/artifacts/scaler.joblib: zapisany obiekt StandardScaler z etapu treningu, potrzebny do tej samej normalizacji cech w trakcie predykcji.
- ml/artifacts/feature_config.json: konfiguracja pipeline cech (np. sample_rate, window_seconds, liczba cech), wykorzystywana jako punkt odniesienia dla zgodnosci danych.

## 6. Struktura repozytorium
- apps/backend: kod backendu (endpointy FastAPI, pipeline inferencyjny, konfiguracja).
- apps/desktop: aplikacja kliencka (GUI, nagrywanie audio, komunikacja HTTP, wizualizacja wynikow).
- ml/research: srodowisko eksperymentow i treningu (notebooki, skrypty przygotowania danych, export artefaktow).
- ml/artifacts: pliki wymagane do inferencji produkcyjnej (model, scaler, konfiguracja cech).
- docs: dokumenty projektowe (architektura, opis technologii, notatki wdrozeniowe).
- infra: konfiguracje uruchomieniowe i konteneryzacyjne (np. Docker Compose, reverse proxy).

## 7. Srodowisko uruchomieniowe
- Python 3.10+ (desktop), Python 3.12+ (backend wg pyproject).
- Menedzer zaleznosci: uv.
- OS: macOS/Windows (desktop), lokalnie lub kontenerowo dla backendu.

Rola srodowiska:
- Python zapewnia wspolna platforme dla desktopu, backendu i warstwy ML.
- uv upraszcza instalacje zaleznosci i uruchamianie komend w obu aplikacjach.
- Podzial na desktop + backend umozliwia lokalne testy end-to-end oraz latwiejsza migracje backendu do kontenera.
