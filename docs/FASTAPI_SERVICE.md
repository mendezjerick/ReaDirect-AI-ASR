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
- `POST /reinforcement/corrections`

`/analyze-audio-file` multipart upload is a future option. Phase 7 supports path-based audio analysis.

## Local Development

Run from the `ReaDirect-AI-ASR` repository root:

```powershell
python scripts\validate_ai_service_startup.py
powershell -ExecutionPolicy Bypass -File scripts\start_ai_service_dev.ps1
```

The service listens on `http://127.0.0.1:8001`.

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8001/health -UseBasicParsing
```

Text smoke test:

```powershell
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

The runtime ASR architecture is Wav2Vec2-only. The active ASR model is `models/wav2vec2-readirect-asr-letters-v2`, with `models/wav2vec2-phoneme` used as supporting acoustic-phonetic evidence for letters and short words. Whisper is removed from runtime routing and is not required by health checks or startup validation.

The FastAPI service is the bridge from Laravel to the AI layer. Laravel remains the official scorer and progression controller. The AI service returns transcript, similarity, phoneme, error-type, feedback, and adaptive recommendation signals.

Audio requests also run lightweight quality validation before ASR. The service reports duration, RMS dBFS, silence/speech ratio, clipping ratio, speech segments, pause metrics, uncertainty reasons, and retry-required metadata. Strict bad-quality cases such as too-short, silent, mostly silent, low-volume, or clipped recordings return `retry_required=true` and `accepted=false` instead of unfairly scoring the learner wrong. See `docs/AUDIO_QUALITY_VALIDATION.md`.

## Deployment Options

- Same server, private localhost port.
- Separate private AI server.
- Docker later.

## Troubleshooting

- If `/health` fails, check the uvicorn process and port.
- If `content_index_loaded=false`, build or copy `data/manifests/content_index.csv`.
- If real ASR fails, verify local Wav2Vec2 folders exist, `TRANSFORMERS_OFFLINE=1`, and `ASR_PROVIDER=wav2vec2_only`.
- If auth fails, check `API_AUTH_ENABLED`, `READIRECT_AI_API_TOKEN`, and the request header.

## Final Runtime Provider

Supported ASR providers:

- `mock`: test/collaborator setup without model loading.
- `wav2vec2_only`: local Hugging Face Wav2Vec2 ASR runtime.

Runtime model paths:

```text
WAV2VEC2_ASR_MODEL_PATH=models/wav2vec2-readirect-asr
WAV2VEC2_PHONEME_MODEL_PATH=models/wav2vec2-phoneme
WAV2VEC2_BASE_ASR_MODEL_PATH=models/wav2vec2-base-960h
ALLOW_WAV2VEC2_BASE_FALLBACK=false
ASR_ARCHITECTURE=wav2vec2_only
```

Letter and word prompts use expected-centric acoustic-phonetic scoring, not raw WER alone. Sentence prompts use the Wav2Vec2 transcript with WER/CER and do not force `displayed_transcript` to `expected_text`.

External training datasets such as Speechocean762 are not required for runtime.
