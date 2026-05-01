# Wav2Vec2 Fine-Tuning

This workflow is training-first. Evaluation is separate, disabled by default, and must not block saving a fine-tuned ASR model.

## Model Scope

- Fine-tune `models/wav2vec2-base-960h` for ASR.
- Save the fine-tuned model to `models/wav2vec2-readirect-asr`.
- Keep `models/wav2vec2-phoneme` unchanged unless a future dataset has safe phoneme-level CTC labels.
- Runtime ASR is Wav2Vec2-only. The fine-tuned Wav2Vec2 model handles letters, words, phrases, sentences, diagnostic assessment, module mastery, and final assessment audio.
- The phoneme model provides supporting evidence for letters and short words.
- Fine-tuning does not replace expected-centric phonetic scoring.

## Local Folders

Expected model folders:

```powershell
models/wav2vec2-base-960h
models/wav2vec2-phoneme
```

Expected dataset folders:

```powershell
external_datasets/librispeech/raw
external_datasets/librispeech/extracted/LibriSpeech/train-clean-100
external_datasets/librispeech/extracted/LibriSpeech/dev-clean
external_datasets/librispeech/extracted/LibriSpeech/test-clean

external_datasets/speechocean/raw
external_datasets/speechocean/extracted
```

The scripts also check the existing `external_datasets/LibriSpeech` and `external_datasets/speechocean762` layouts.

## Extraction

LibriSpeech archives can be extracted with:

```powershell
tar -xzf external_datasets/librispeech/raw/train-clean-100.tar.gz -C external_datasets/librispeech/extracted
tar -xzf external_datasets/librispeech/raw/dev-clean.tar.gz -C external_datasets/librispeech/extracted
tar -xzf external_datasets/librispeech/raw/test-clean.tar.gz -C external_datasets/librispeech/extracted
```

SpeechOcean extraction depends on the archive you have. Keep raw archives under `external_datasets/speechocean/raw` or `external_datasets/speechocean762/raw` and extract under the matching `extracted` folder.

## Install

```powershell
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, use the official PyTorch selector if the default wheel is CPU-only.

## Validate Setup

```powershell
python scripts/validate_training_setup.py
```

Evaluation-only packages such as `jiwer`, `evaluate`, and `scikit-learn` warn instead of blocking training.

## Prepare Manifests

```powershell
python scripts/prepare_librispeech_manifest.py
python scripts/prepare_speechocean_manifest.py
python scripts/build_wav2vec2_training_manifest.py
python scripts/validate_training_manifest.py external_datasets/manifests/readirect_train_mixed.jsonl
```

LibriSpeech parsing reads `.flac` files and matching `.trans.txt` transcript files. SpeechOcean parsing first detects the known Speechocean762 `train.json` and `test.json` layout, preserving pronunciation score metadata when present. If the SpeechOcean layout is unknown, the script scans common transcript files and writes a report to `outputs/training/speechocean_manifest_report.txt`.

The minimum training manifest is:

```powershell
external_datasets/manifests/readirect_train_mixed.jsonl
```

Validation and test manifests are optional for training.

## Smoke Test Training

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --smoke-test --no-eval --max-train-samples 50
```

This loads the local base model, reads a tiny train subset, loads audio, tokenizes labels, runs a few steps, saves a checkpoint, and saves the final model.

## Full Training

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --no-eval
```

Staged options:

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --stage librispeech --no-eval
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --stage speechocean --no-eval
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --stage mixed --no-eval
```

Default config uses a mixed-manifest strategy with SpeechOcean weighted higher than LibriSpeech. LibriSpeech is an anchor, not the main adaptation target.

## Evaluation Later

Run evaluation manually after training:

```powershell
python scripts/evaluate_wav2vec2_readirect_asr.py --model models/wav2vec2-readirect-asr --manifest external_datasets/manifests/readirect_valid_mixed.jsonl
```

Evaluation output is saved to:

```powershell
outputs/evaluation/wav2vec2_readirect_full_eval.json
```

Hard-case evaluation is only a template at:

```powershell
external_datasets/manifests/readirect_hard_cases.example.jsonl
```

It is not part of the training loop.

## API Model Paths

Runtime ASR routing uses:

```powershell
WAV2VEC2_ASR_MODEL_PATH=models/wav2vec2-readirect-asr
WAV2VEC2_PHONEME_MODEL_PATH=models/wav2vec2-phoneme
WAV2VEC2_BASE_ASR_MODEL_PATH=models/wav2vec2-base-960h
ALLOW_WAV2VEC2_BASE_FALLBACK=false
ASR_ARCHITECTURE=wav2vec2_only
```

These are in `.env.example`. Do not hardcode absolute paths. The base model is used only when fallback is explicitly enabled.

## Generated Artifacts

Datasets, checkpoints, outputs, and large model files are ignored by Git:

```powershell
external_datasets/
checkpoints/
outputs/
models/wav2vec2-readirect-asr/
*.pt
*.bin
*.safetensors
```

## Limitations

- LibriSpeech is general clean English speech, not pronunciation-specific learner speech.
- SpeechOcean structure and license terms must be verified locally.
- Wav2Vec2 fine-tuning does not replace expected-centric phonetic scoring.
- Q vs U and similar letter cases still require dedicated phoneme and expected-answer logic.
- Sentence runtime scoring uses the Wav2Vec2 transcript plus WER/CER; letter and word correction rules are not applied to full sentences.
