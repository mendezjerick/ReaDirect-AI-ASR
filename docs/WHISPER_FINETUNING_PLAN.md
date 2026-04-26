# Whisper Fine-Tuning Plan

Phase 10 implements the guarded fine-tuning pipeline. Training is still manual and must be started explicitly with `--run`.

## Recommended Starting Point

Start with `openai/whisper-base.en`.

Use a small, controlled experiment before trying larger models.

## Training Approach

Planned stack:

- Hugging Face Transformers
- Whisper processor/tokenizer
- JSONL files from `data/processed/whisper_finetune/`
- Manual transcripts as training targets

Audio loading uses `librosa` by default:

```yaml
audio_loading_backend: librosa
```

The trainer intentionally avoids `datasets.Audio` so Windows fine-tuning does not depend on TorchCodec or full-shared FFmpeg DLL discovery. The preprocessing step loads audio with `librosa.load(audio_path, sr=16000, mono=True)` and feeds the array directly to `WhisperProcessor`.

Config template:

```text
configs/whisper_finetune_config.yaml
```

Dry-run safety:

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --dry-run
```

Actual training:

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --run
```

Training-time evaluation is disabled by default to avoid generation-config failures during checkpoints:

```yaml
run_eval_during_training: false
evaluation_strategy: "no"
```

The trainer saves checkpoints and the final model normally. Run evaluation separately after training completes.

Evaluation:

```powershell
python scripts/evaluate_finetuned_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --test-jsonl data/processed/whisper_finetune/test.jsonl --output reports/finetuned_whisper_eval.md --metrics-json reports/finetuned_whisper_metrics.json
```

Optional conversion:

```powershell
python scripts/convert_whisper_to_faster_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --output-dir model_artifacts/readirect-whisper-base-en-v1-ct2 --quantization int8_float16 --dry-run
```

## Hardware Notes

An RTX 3060 with 12GB VRAM can likely handle `tiny.en` or `base.en`, and may handle `small.en` with careful settings.

Recommended settings:

- `fp16: true`
- small per-device batch size
- gradient accumulation
- regular evaluation
- checkpoint retention outside Git

## Artifact Handling

Local outputs should go under `model_artifacts/`.

Do not commit checkpoints, model weights, optimizer states, downloaded model cache, or converted CTranslate2 exports.

Share final approved models through secure external storage or a private model registry.

See `docs/MANUAL_FINETUNING_STEPS.md` for Windows/RTX 3060 troubleshooting.

## Optional Conversion

After a model is trained and validated, it may be converted for faster-whisper/CTranslate2 inference:

```powershell
python scripts/convert_whisper_to_faster_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --output-dir model_artifacts/readirect-whisper-base-en-v1-ct2 --quantization int8_float16 --dry-run
```

Converted model artifacts remain local and are not committed to GitHub.

## Current Fine-Tuning Result

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| WER | 0.494 | 0.396 |
| CER | 0.324 | 0.197 |
| Exact Match Rate | 0.340 | 0.304 |

Fine-tuning improved WER and CER. Exact match remains strict, so the fine-tuned ASR layer should be combined with ReaDirect's explainable reading-analysis engine.
