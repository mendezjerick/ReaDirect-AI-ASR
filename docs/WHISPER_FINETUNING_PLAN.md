# Whisper Fine-Tuning Plan

This is a future implementation plan. Phase 9 only prepares data and decision reports.

## Recommended Starting Point

Start with `openai/whisper-base.en`.

Use a small, controlled experiment before trying larger models.

## Training Approach

Planned stack:

- Hugging Face Transformers
- Whisper processor/tokenizer
- JSONL files from `data/processed/whisper_finetune/`
- Manual transcripts as training targets

Config template:

```text
configs/whisper_finetune_config.yaml
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

## Future Conversion

After a model is trained and validated, it may be converted for faster-whisper/CTranslate2 inference later. That is not part of Phase 9.
