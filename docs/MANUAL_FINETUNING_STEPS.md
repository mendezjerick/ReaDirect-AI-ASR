# Manual Fine-Tuning Steps

Phase 10 creates the training pipeline, but training only starts when you manually run the command with `--run`.

## Step 1: Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

## Step 2: Check GPU/CUDA

```powershell
python scripts/check_training_environment.py
```

Quick direct check:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is false, PyTorch may be CPU-only. Install a CUDA-enabled PyTorch build using the official PyTorch selector, then restart PowerShell.

## Audio Loading Note

The fine-tuning pipeline is configured with:

```yaml
audio_loading_backend: librosa
```

This avoids Hugging Face `datasets.Audio`, TorchCodec, and FFmpeg shared-DLL issues on Windows. Audio is loaded manually with:

```python
librosa.load(audio_path, sr=16000, mono=True)
```

Basic fine-tuning should not require TorchCodec or FFmpeg as long as `librosa` can read the dataset audio files.

## Step 3: Prepare Fine-Tuning Dataset

```powershell
python scripts/prepare_whisper_finetune_dataset.py --manifest data/manifests/speechocean762_manifest.csv --output-dir data/processed/whisper_finetune
```

## Step 4: Dry Run

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --dry-run
```

This validates config, JSONL files, and CUDA status. It does not download a model or train.

Training-time evaluation is disabled by default:

```yaml
run_eval_during_training: false
evaluation_strategy: "no"
```

This prevents Transformers/Whisper generation-config errors from stopping a long training run. Train first, then evaluate separately after the model is saved.

## Step 5: Start Actual Training Manually

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --run
```

## Step 6: Evaluate Fine-Tuned Model

```powershell
python scripts/evaluate_finetuned_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --test-jsonl data/processed/whisper_finetune/test.jsonl --output reports/finetuned_whisper_eval.md --metrics-json reports/finetuned_whisper_metrics.json
```

## Step 7: Optional Conversion Dry Run

```powershell
python scripts/convert_whisper_to_faster_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --output-dir model_artifacts/readirect-whisper-base-en-v1-ct2 --quantization int8_float16 --dry-run
```

## Step 8: Optional Actual Conversion

```powershell
python scripts/convert_whisper_to_faster_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --output-dir model_artifacts/readirect-whisper-base-en-v1-ct2 --quantization int8_float16
```

## Troubleshooting

If CUDA is false:

- Reinstall PyTorch with CUDA support from the official PyTorch selector.
- Restart PowerShell after reinstalling.
- Run `python scripts/check_training_environment.py` again.

If out of memory:

- Set `per_device_train_batch_size: 1`.
- Set `gradient_accumulation_steps: 16`.
- Keep `fp16: true`.
- Keep `gradient_checkpointing: true`.
- Try `openai/whisper-tiny.en`.

If training is too slow:

- Reduce `max_steps`.
- Use `openai/whisper-tiny.en` for the first test.
- Add `--limit-train` and `--limit-validation` for debugging.

If model download is slow:

- Wait for the first Hugging Face download to finish.
- Optional: authenticate with Hugging Face if needed.
- Do not commit cache or downloaded model files.

If checkpoints become too large:

- `save_total_limit` is already set to `2`.
- Delete old checkpoints if needed.
- Never commit `model_artifacts/`.

If TorchCodec or FFmpeg errors appear:

- Make sure `configs/whisper_finetune_config.yaml` has `audio_loading_backend: librosa`.
- Pull the latest `training/train_whisper.py` that avoids `datasets.Audio`.
- Rerun the dry run.
- Do not change your working CUDA PyTorch install just to fix TorchCodec.

## Current Result Summary

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| WER | 0.494 | 0.396 |
| CER | 0.324 | 0.197 |
| Exact Match Rate | 0.340 | 0.304 |

The fine-tuned model improves overall transcript closeness, but exact match remains challenging. Use the fine-tuned model together with ReaDirect's reading-analysis engine.
