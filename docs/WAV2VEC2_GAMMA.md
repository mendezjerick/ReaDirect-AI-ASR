# Wav2Vec2 Gamma

Gamma continues the clean ASR line from the final Beta model, which was saved after loading Beta's best validation-WER checkpoint.

## Start Checkpoint

```text
models/asr/beta
```

Override only with a complete final/best Beta model directory:

```powershell
$env:BETA_CHECKPOINT_PATH = "C:\path\to\validated\beta"
```

Gamma does not restart from Alpha or base-960h. Base-960h is loaded only for a tokenizer, processor, delimiter, and CTC-head compatibility safety check.

## Training Data

- Full local GigaSpeech S train split: 230,068 rows before duration filtering
- ReaDirect isolated letters: 780 unique rows
- SpeechOcean762 training rows: 0

The default configuration uses:

```powershell
$env:GIGASPEECH_S_MAX_ROWS = "0"
$env:READIRECT_LETTERS_REPEAT_FACTOR = "1"
```

`0` means all GigaSpeech S rows. Letters are included once and are not oversampled.

GigaSpeech S is read only from the existing local parquet directory:

```text
external_datasets/gigaspeech_s_parquet/parquet-data/s
```

The Gamma scripts do not access the Hugging Face Hub and do not download data.

Excluded from training: SpeechOcean762, Common Voice, L2-ARCTIC, PLD, MyST, TED-LIUM, The People's Speech, child-speech datasets, and all other datasets. Beam search, expected-centric correction, boundary repair, GOP, and production scoring are also excluded.

## Training Defaults

- 1 epoch by default
- Maximum allowed: 2 epochs
- Shared validation every epoch
- Save every epoch
- Select best checkpoint by validation WER
- Load best model before saving `models/asr/gamma`
- Early stopping supported
- Greedy CTC only

SpeechOcean remains in the historical shared validation set only. It is never used for Gamma training.

## Paths

```powershell
$env:BETA_CHECKPOINT_PATH = "models/asr/beta"
$env:GIGASPEECH_S_PATH = "external_datasets/gigaspeech_s_parquet/parquet-data/s"
$env:GIGASPEECH_CACHE_DIR = "external_datasets/hf_cache"
$env:GIGASPEECH_S_MAX_ROWS = "0"
$env:READIRECT_LETTERS_ROOT = "external_datasets/readirect_letters"
$env:READIRECT_LETTERS_REPEAT_FACTOR = "1"
```

## Preflight

```powershell
python scripts/validate_wav2vec2_gamma_setup.py --config configs/wav2vec2_gamma.yaml --decode-sample
```

## Manual Training

```powershell
python scripts/train_wav2vec2_gamma.py --config configs/wav2vec2_gamma.yaml
```

## Manual Evaluation

```powershell
python scripts/evaluate_wav2vec2_gamma.py --config configs/wav2vec2_gamma.yaml --model models/asr/gamma --split validation
```

Evaluation reports shared WER/CER, SpeechOcean WER/CER, isolated-letter accuracy, and comparison metrics for Base, V1, V2, Alpha, Beta, and Gamma.

GigaSpeech-only validation uses the existing cached 6,750-row validation split:

```powershell
python scripts/evaluate_wav2vec2_gamma.py --config configs/wav2vec2_gamma.yaml --model models/asr/gamma --split gigaspeech-validation
```

## Outputs

```text
models/asr/gamma/
checkpoints/asr/gamma/
reports/asr/gamma/
logs/asr/gamma/
```

Only a hard-case example template currently exists. Hard-case evaluation remains disabled until a real manifest is configured.
