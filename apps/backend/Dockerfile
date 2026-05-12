FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/pyproject.toml /app/apps/backend/pyproject.toml
COPY apps/backend/src /app/apps/backend/src
COPY ml/artifacts /app/ml/artifacts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /app/apps/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--app-dir", "/app/apps/backend/src", "--host", "0.0.0.0", "--port", "8000"]
