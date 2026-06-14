# Wav2Vec2 Alpha

Alpha is the first model in the clean Greek-letter ASR training line. The V1/V2 files remain historical and are not modified or used as an Alpha checkpoint.

## Scope

- Base checkpoint: `facebook/wav2vec2-base-960h`
- Local base checkpoint: `models/wav2vec2-base-960h`
- Decoder: raw greedy CTC only
- Metrics: WER, CER, and ReaDirect isolated-letter exact accuracy
- Not included: beam search, expected-centric correction, boundary repair, or GOP scoring

## Included Datasets

- GigaSpeech XS from the existing Hugging Face cache
- SpeechOcean762 as a capped support dataset
- ReaDirect isolated A-Z recordings

The default train mix uses all 9,389 GigaSpeech XS train rows, a deterministic sample of 1,000 SpeechOcean rows, and all 780 ReaDirect letter train rows. This is approximately 84% GigaSpeech, 9% SpeechOcean, and 7% letters. Letters are not oversampled.

## Excluded Datasets

Common Voice, L2-ARCTIC, PLD, MyST, TED-LIUM, The People's Speech, child-speech datasets, LibriSpeech training data, and every dataset not listed above are excluded.

## Paths

Alpha artifacts are isolated:

```text
models/asr/alpha/
checkpoints/asr/alpha/
reports/asr/alpha/
logs/asr/alpha/
```

Defaults can be changed in `configs/wav2vec2_alpha.yaml` or with:

```powershell
$env:GIGASPEECH_CACHE_DIR = "C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-AI-ASR\external_datasets\hf_cache"
$env:WAV2VEC2_ALPHA_BASE_MODEL_PATH = "models/wav2vec2-base-960h"
$env:SPEECHOCEAN_TRAIN_MANIFEST = "external_datasets/manifests/speechocean_train.jsonl"
$env:SPEECHOCEAN_VALID_MANIFEST = "external_datasets/manifests/speechocean_valid.jsonl"
$env:SPEECHOCEAN_TEST_MANIFEST = "external_datasets/manifests/speechocean_test.jsonl"
$env:READIRECT_LETTERS_ROOT = "external_datasets/readirect_letters"
```

The GigaSpeech loader reads the cached Arrow shards directly. It does not call the Hub and does not re-download GigaSpeech.

## FFmpeg And TorchCodec

Hugging Face `Audio` decoding accesses `audio["array"]` through TorchCodec. Before training:

1. Install a TorchCodec version compatible with the installed PyTorch.
2. Install an FFmpeg shared build, not only a standalone static executable.
3. Put the shared-build `bin` directory on `PATH`, or set:

```powershell
$env:FFMPEG_BIN_DIR = "C:\Users\Lost\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build-shared\bin"
```

The directory must contain FFmpeg DLLs such as `avcodec-*.dll`, `avutil-*.dll`, and related libraries.
The Alpha scripts also auto-detect the WinGet `Gyan.FFmpeg.Shared` installation when `FFMPEG_BIN_DIR` is unset.

Validate imports, paths, cached split counts, and one real GigaSpeech decode:

```powershell
.\.venv\Scripts\python.exe scripts/validate_wav2vec2_alpha_setup.py --config configs/wav2vec2_alpha.yaml --decode-sample
```

This validation does not start training.

## Manual Training

Run manually from `ReaDirect-AI-ASR`:

```powershell
.\.venv\Scripts\python.exe scripts/train_wav2vec2_alpha.py --config configs/wav2vec2_alpha.yaml
```

Training-time evaluation is disabled. Alpha starts from the local copy of `facebook/wav2vec2-base-960h`, not V1 or V2.

## Manual Evaluation

Evaluate the combined validation sources with raw greedy CTC decoding:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_wav2vec2_alpha.py --config configs/wav2vec2_alpha.yaml --model models/asr/alpha --split validation
```

The result is written to `reports/asr/alpha/alpha_validation_evaluation.json`.

The repository currently contains only `readirect_hard_cases.example.jsonl`, which is a template rather than a real hard-case set. Hard-case evaluation remains disabled until `evaluation.hard_case_manifest` points to a real manifest.
