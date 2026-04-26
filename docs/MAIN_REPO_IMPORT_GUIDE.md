# Main ReaDirect Repository Import Guide

The AI service remains a separate repository and service. Do not copy external training datasets into the main Laravel repository.

Do not copy:

- Speechocean762 archive
- extracted Speechocean762 audio
- training manifests
- training JSONL files
- model checkpoints
- raw external datasets

Runtime model artifacts stay in the AI service deployment path:

```text
ReaDirect-AI-ASR/model_artifacts/readirect-whisper-base-en-v1-hf/
```

Optional converted model:

```text
ReaDirect-AI-ASR/model_artifacts/readirect-whisper-base-en-v1-ct2/
```

Suggested main repo destinations:

- Integration docs: `ReaDirect/docs/ai-service/`
- Laravel `.env` values: `ReaDirect/.env` and `ReaDirect/.env.example`
- Enriched content ZIP: `ReaDirect/content-bank/import/readirect-enriched-content.zip`
- Reviewed enriched CSVs: `ReaDirect/database/seed-data/readirect/enriched/`

Suggested future Laravel code locations:

- `ReaDirect/app/Services/AI/ReadirectAIService.php`
- `ReaDirect/config/readirect_ai.php`
- `ReaDirect/tests/Feature/AI/`
- admin debug pages as needed

Laravel should call:

```text
http://127.0.0.1:8001/analyze-audio
http://127.0.0.1:8001/analyze-text
http://127.0.0.1:8001/recommend-next
```
