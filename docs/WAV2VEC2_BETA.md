# Wav2Vec2 Beta

Beta continues the clean Greek-letter ASR line from the externally validated Alpha model. It does not use V1/V2 and does not restart from base-960h.

## Start Checkpoint

Alpha's step checkpoints were created without validation, so they do not contain a recorded best validation checkpoint. The validated final Alpha artifact is therefore the default:

```text
models/asr/alpha
```

Override it only with a complete validated model directory:

```powershell
$env:ALPHA_CHECKPOINT_PATH = "C:\path\to\validated\alpha"
```

Beta verifies the Alpha tokenizer mapping and CTC head shape against `models/wav2vec2-base-960h` before training.

## Datasets

Default training mix:

- GigaSpeech S: 12,000 deterministic rows
- SpeechOcean762: 1,000 deterministic rows
- ReaDirect isolated letters: all 780 unique rows

Approximate ratio: 87.1% / 7.3% / 5.7%. Letters are not oversampled.

GigaSpeech S is loaded only from existing local parquet files under:

```text
external_datasets/gigaspeech_s_parquet/parquet-data/s
```

No Hub download is performed by the Beta scripts.

Excluded: Common Voice, L2-ARCTIC, PLD, MyST, TED-LIUM, The People's Speech, child-speech datasets, beam search, expected-centric correction, boundary repair, GOP, and production scoring logic.

## Training Behavior

- Default: 2 epochs
- Maximum permitted by config: 3 epochs
- Shared validation every epoch
- Checkpoint every epoch
- Best checkpoint selected by validation WER
- Best model loaded before saving `models/asr/beta`
- Early stopping patience: 2 validation epochs
- Greedy CTC only

The epoch validation set is the same 250 SpeechOcean and 260 ReaDirect letter samples used for the shared model comparison.

## Paths

```powershell
$env:ALPHA_CHECKPOINT_PATH = "models/asr/alpha"
$env:GIGASPEECH_S_PATH = "external_datasets/gigaspeech_s_parquet/parquet-data/s"
$env:GIGASPEECH_CACHE_DIR = "external_datasets/hf_cache"
$env:SPEECHOCEAN_TRAIN_MANIFEST = "external_datasets/manifests/speechocean_train.jsonl"
$env:READIRECT_LETTERS_ROOT = "external_datasets/readirect_letters"
```

## Preflight

```powershell
.\.venv\Scripts\python.exe scripts/validate_wav2vec2_beta_setup.py --config configs/wav2vec2_beta.yaml --decode-sample
```

## Manual Training

```powershell
.\.venv\Scripts\python.exe scripts/train_wav2vec2_beta.py --config configs/wav2vec2_beta.yaml
```

## Manual Evaluation

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_wav2vec2_beta.py --config configs/wav2vec2_beta.yaml --model models/asr/beta --split validation
```

The evaluation reports shared WER/CER, SpeechOcean WER/CER, and isolated-letter accuracy.

## Outputs

```text
models/asr/beta/
checkpoints/asr/beta/
reports/asr/beta/
logs/asr/beta/
```

The repository contains only a hard-case example template, not a real hard-case manifest. Hard-case evaluation remains disabled until `evaluation.hard_case_manifest` is configured.

