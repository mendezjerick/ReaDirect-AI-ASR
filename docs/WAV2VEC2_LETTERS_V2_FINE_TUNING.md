# Wav2Vec2 Letters v2 Fine-Tuning

This workflow is for continued fine-tuning only. Evaluation and metrics are separate from training.

## Purpose

The v2 model is planned to improve isolated English letter recognition while preserving word and sentence ASR performance.

## Model Paths

Base model:

```powershell
models/wav2vec2-readirect-asr
```

Output model:

```powershell
models/wav2vec2-readirect-asr-letters-v2
```

The output path is separate so the active v1 model is not overwritten.

## Training Mix

The v2 training manifest targets:

- 50% ReaDirect custom letters
- 30% SpeechOcean
- 20% LibriSpeech

The ratio applies to training data. Validation and test manifests are combined for later evaluation only.

## Training-First Rule

Training must be able to load the training manifest, train, save checkpoints, save the final v2 model, write training logs/summary, and exit without running evaluation.

Default behavior:

- No WER during training.
- No CER during training.
- No hard-case evaluation during training.
- No model comparison during training.
- Evaluation during training disabled by default.

If light validation is ever enabled explicitly and fails, it must not prevent the saved model from existing. The recommended v2 commands use `--no-eval`.

## Warnings

- Do not overwrite `models/wav2vec2-readirect-asr`.
- Do not activate v2 before separate evaluation.
- Do not train `models/wav2vec2-phoneme`.
- Do not reintroduce Whisper.
- Do not switch runtime paths as part of training.

## Commands

Validate inputs:

```powershell
python scripts/validate_wav2vec2_v2_training_inputs.py
```

Build the v2 mixed manifest:

```powershell
python scripts/build_wav2vec2_letters_v2_manifest.py --letters-ratio 0.50 --speechocean-ratio 0.30 --librispeech-ratio 0.20 --seed 42
```

Smoke test training:

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_letters_v2.yaml --smoke-test --no-eval --require-cuda --max-train-samples 50
```

Full training:

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_letters_v2.yaml --no-eval --require-cuda
```

Evaluation later:

Do not include evaluation in the training command. Evaluation must be run separately after training with a separate command and only after the v2 model has been saved.

## Generated Artifacts

Training writes:

```powershell
checkpoints/wav2vec2-readirect-asr-letters-v2
models/wav2vec2-readirect-asr-letters-v2
outputs/training/wav2vec2_letters_v2_training_summary.json
outputs/training/wav2vec2_letters_v2_training_log.txt
models/wav2vec2-readirect-asr-letters-v2/readirect_model_metadata.json
```

The model metadata marks:

- `evaluation_completed: false`
- `wer: null`
- `cer: null`
- `runtime_active: false`

Do not invent WER/CER. Run separate evaluation before considering runtime promotion.
