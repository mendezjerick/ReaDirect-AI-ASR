# API

FastAPI service code lives here. The production ASR runtime is Wav2Vec2-only: `models/asr/epsilon` for transcription and `models/wav2vec2-phoneme` for supporting phoneme evidence. Normal transcription defaults to CTC beam search with the configured KenLM model. No-LM beam fallback must be explicitly enabled; greedy decoding is a debug option. Previous ASR models remain available for historical evaluation only. `MockASR` remains available for tests and collaborator setup without model loading.

Do not put API keys, `.env` files, learner audio, or private request logs in this directory.

Run locally:

```powershell
uvicorn api.main:app --reload --port 8001
```
