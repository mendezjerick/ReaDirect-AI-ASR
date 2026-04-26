# FastAPI Service

The FastAPI service is the Laravel-facing API for ReaDirect AI/ASR analysis.

## Architecture

- `api/main.py`: app, routes, CORS.
- `api/schemas.py`: request/response contracts.
- `api/service.py`: orchestration.
- `api/dependencies.py`: config, ASR provider, CMUdict, content repository.
- `api/security.py`: optional token auth.
- `api/errors.py`: safe error helpers.

## Endpoints

- `GET /health`
- `GET /version`
- `POST /analyze-text`
- `POST /analyze-audio`
- `POST /content-item`
- `POST /analyze-content-item`

`/analyze-audio-file` multipart upload is a future option. Phase 7 supports path-based audio analysis.

## Local Development

```powershell
uvicorn api.main:app --reload --port 8001
python scripts/test_api_analysis.py --mode text --expected-text cat --actual-text cap --accepted-answer cat --debug
```

## Laravel `.env` Example

```text
READIRECT_AI_ENABLED=true
READIRECT_AI_BASE_URL=http://127.0.0.1:8001
READIRECT_AI_API_TOKEN=
READIRECT_AI_TIMEOUT_SECONDS=60
```

## Security

Local development can use:

```text
API_AUTH_ENABLED=false
```

Production should use:

```text
API_AUTH_ENABLED=true
READIRECT_AI_API_TOKEN=<server-secret>
```

Laravel should send `X-ReaDirect-AI-Token`.

Students should never call this service directly. Keep it private and call it server-to-server from Laravel.

Fine-tuned Whisper models are not required for the API, but a reviewed fine-tuned artifact can later be configured as the ASR provider. Keep using `mock` or `faster_whisper` provider settings until a reviewed model artifact exists.

The FastAPI service is the bridge from Laravel to the AI layer. Laravel remains the official scorer and progression controller. The AI service returns transcript, similarity, phoneme, error-type, feedback, and adaptive recommendation signals.

## Deployment Options

- Same server, private localhost port.
- Separate private AI server.
- Docker later.

## Troubleshooting

- If `/health` fails, check the uvicorn process and port.
- If `content_index_loaded=false`, build or copy `data/manifests/content_index.csv`.
- If real ASR fails, verify `faster-whisper` is installed and `ASR_PROVIDER=faster_whisper`.
- If auth fails, check `API_AUTH_ENABLED`, `READIRECT_AI_API_TOKEN`, and the request header.

## Final Runtime Providers

Supported ASR providers:

- `mock`: test/collaborator setup without model loading.
- `faster_whisper_pretrained`: pretrained faster-whisper model such as `base.en`.
- `faster_whisper_local`: converted CTranslate2/faster-whisper model from `model_artifacts/readirect-whisper-base-en-v1-ct2/`.
- `hf_whisper_local`: local Hugging Face fine-tuned model from `model_artifacts/readirect-whisper-base-en-v1-hf/`.

External training datasets such as Speechocean762 are not required for runtime.
