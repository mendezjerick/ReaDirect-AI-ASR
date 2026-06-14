# Wav2Vec2 Shared Benchmark

This benchmark evaluates every model on the same deterministic five-source set.
Each source contributes 250 rows, preventing large corpora from dominating the
overall score.

## Sources

- LibriSpeech `test-clean`: clean held-out source
- GigaSpeech official `validation`: clean held-out source
- ReaDirect isolated letters `test`: clean held-out source
- SpeechOcean762 `test`: clean held-out source
- SLR83 deterministic diagnostic sample

The primary clean metrics macro-average LibriSpeech, GigaSpeech, ReaDirect
letters, and SpeechOcean so each clean source receives equal weight. A
five-source diagnostic macro includes SLR83. Per-source WER/CER and ReaDirect
isolated-letter accuracy are also reported.

## Fairness Limitation

Delta trained on all SLR83 rows, so its SLR83 result is training-contaminated.
The evaluator reports SLR83 diagnostically for every model but excludes SLR83
from every model's `clean_macro_wer` and `clean_macro_cer`. This keeps the
primary ranking on the same four clean sources. A fully clean five-source
ranking requires retraining Delta with a speaker-held-out SLR83 split.

Gamma trained on all local GigaSpeech S training rows. For that reason, this
benchmark requires the official GigaSpeech validation split and never uses
local S training rows for evaluation.

## Prepare GigaSpeech Validation

The validation parquet was removed during cache cleanup. Download only that
split manually:

```powershell
hf auth login
python scripts/download_gigaspeech_validation.py
```

If Hugging Face is already authenticated on this machine, the login command is
not needed.

## Build Benchmark

```powershell
python scripts/build_wav2vec2_shared_benchmark.py --config configs/wav2vec2_shared_benchmark.yaml
```

## Evaluate Models

```powershell
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name base
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name v1
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name v2
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name alpha
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name beta
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name gamma
python scripts/evaluate_wav2vec2_shared_benchmark.py --model-name delta
```

## Compare

```powershell
python scripts/compare_wav2vec2_shared_benchmark.py
```

Outputs are saved under:

```text
reports/asr/shared_benchmark/
```
