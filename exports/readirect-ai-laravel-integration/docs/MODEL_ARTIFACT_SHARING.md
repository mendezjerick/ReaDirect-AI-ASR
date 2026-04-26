# Model Artifact Sharing

Collaborators do not need the full dataset and should not retrain by default.

One approved training run can produce a final model artifact. Share that artifact outside Git using one of:

- Google Drive
- school OneDrive
- private storage
- private Hugging Face repository
- private model registry

Collaborators place downloaded models under:

```text
model_artifacts/
```

The main Laravel app does not need training files. Laravel calls the FastAPI AI service, and the AI service loads whichever ASR provider/model is configured.

Do not commit:

- checkpoints
- model weights
- optimizer states
- CTranslate2 converted model folders
- Hugging Face cache files
- generated training logs

For deployment, place the fine-tuned Hugging Face model folder at:

```text
ReaDirect-AI-ASR/model_artifacts/readirect-whisper-base-en-v1-hf/
```

If a CTranslate2/faster-whisper model is used, place it at:

```text
ReaDirect-AI-ASR/model_artifacts/readirect-whisper-base-en-v1-ct2/
```
