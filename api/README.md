# API

FastAPI service code lives here. In Phase AI-1 the service uses `MockASR` and does not require real audio decoding, GPU access, or model downloads.

Do not put API keys, `.env` files, learner audio, or private request logs in this directory.

Run locally:

```powershell
uvicorn api.main:app --reload --port 8001
```

