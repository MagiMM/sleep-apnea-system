# Sleep Apnea System

Monorepo for the sleep apnea detection product.

## Structure
- `apps/backend` - FastAPI inference service
- `apps/desktop` - desktop client for audio capture and 10 s windowing
- `ml/research` - notebook-based model research and experimentation
- `ml/artifacts` - exported model files, scaler, and feature config
- `shared` - shared schemas and contract notes
- `infra` - Docker and reverse proxy configuration
- `docs` - architecture and implementation notes

## First step
Start with the backend service, because it defines the model contract and the inference pipeline.
