# API

FastAPI service code lives here. The production ASR runtime is Wav2Vec2-only: `models/wav2vec2-readirect-asr` for transcription and `models/wav2vec2-phoneme` for supporting phoneme evidence. `MockASR` remains available for tests and collaborator setup without model loading.

Do not put API keys, `.env` files, learner audio, or private request logs in this directory.

Run locally:

```powershell
uvicorn api.main:app --reload --port 8001
```
