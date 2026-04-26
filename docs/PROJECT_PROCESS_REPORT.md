# ReaDirect-AI-ASR Project Process Report

## 1. Data Preparation

The ReaDirect-AI-ASR repository was structured as a separate Python AI layer for speech recognition, reading analysis, content enrichment, adaptive tutoring, and Laravel-facing service integration. Data preparation separated raw datasets, processed files, manifests, content-bank files, enriched content, model artifacts, and reports.

The main working folders are:

- `external_datasets/`
- `data/raw/`
- `data/processed/`
- `data/manifests/`
- `content_bank/`
- `content_bank_enriched/`
- `model_artifacts/`
- `reports/`

This separation makes the workflow reproducible while keeping large datasets, generated artifacts, and private data out of GitHub.

## 2. Data Acquisition & Collection

The active public speech dataset is Speechocean762, stored locally under `external_datasets/speechocean762/`. It was used for baseline ASR evaluation and Whisper fine-tuning. The repository also uses CMUdict under `external_datasets/cmudict/` for word-to-phoneme mapping.

The ReaDirect curated CSV content bank provides expected reading items, module activities, assessment content, and accepted answers. It is used to build `data/manifests/content_index.csv` and enriched content exports.

L2-ARCTIC was excluded from the active workflow because of non-commercial licensing concerns. PF-STAR was excluded because it requires access/request approval.

## 3. Data Cleaning & Preprocessing

The Speechocean762 archive was inspected, extracted, and converted into a unified manifest. Audio paths, transcripts, durations, dataset source, speaker metadata, and scoring annotations were normalized where available.

Relevant preprocessing scripts include:

- `scripts/inspect_speechocean762.py`
- `scripts/extract_speechocean762.py`
- `scripts/build_speechocean762_manifest.py`
- `scripts/validate_manifest.py`
- `scripts/prepare_whisper_finetune_dataset.py`

The Whisper training preparation creates:

- `data/processed/whisper_finetune/train.jsonl`
- `data/processed/whisper_finetune/validation.jsonl`
- `data/processed/whisper_finetune/test.jsonl`

These files are generated locally and ignored by Git.

## 4. Methodology

The methodology combines neural ASR and explainable reading-analysis logic:

```text
audio -> ASR transcription -> normalization -> expected-answer comparison
-> phoneme mapping -> error detection -> feedback hints -> adaptive recommendation
```

Whisper/faster-whisper handles speech recognition. CMUdict supports phoneme analysis. ReaDirect-specific scoring modules perform expected-answer matching, transcript-derived phoneme comparison, error-type detection, feedback hint mapping, and adaptive skill-signal generation.

Laravel remains the official scorer and progression controller. This AI service provides structured analysis signals.

## 5. Results

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| WER | 0.494 | 0.396 |
| CER | 0.324 | 0.197 |
| Exact Match Rate | 0.340 | 0.304 |

The fine-tuned Whisper model improved WER and CER compared with the pretrained baseline. Exact match decreased slightly, showing that strict transcript equality is still difficult and should not be the sole basis for learner scoring.

## 6. Interpretation

Fine-tuning improved the ASR layer, but ReaDirect should still use a hybrid AI approach. The final learner analysis should combine:

- ASR transcript
- expected-answer comparison
- accepted-answer matching
- phoneme-aware comparison
- error-type detection
- feedback hints
- adaptive recommendation signals

This design supports a reading-tutor workflow rather than a simple correct/incorrect quiz checker.

## 7. Limitations

- The fine-tuned model was trained and evaluated on Speechocean762, not actual ReaDirect learner recordings.
- Phoneme analysis is derived from ASR text, not acoustic phoneme recognition.
- Short-word ASR errors may still occur.
- Exact match remains strict and can understate partial correctness.
- Future validation should use consented pilot learner recordings.

## 8. Repository Reproducibility Notes

GitHub includes source code, scripts, configs, tests, documentation, and safe placeholders. GitHub excludes dataset archives, extracted audio, generated manifests, generated reports, model checkpoints, model artifacts, `.env`, private learner data, and real learner audio.

To reproduce the workflow locally, install dependencies, place Speechocean762 in `external_datasets/speechocean762/raw/`, build the manifest, run baseline ASR, prepare Whisper JSONL files, train manually, and evaluate the fine-tuned model.

For Laravel deployment, external training datasets are not imported into the main repository. Runtime only needs the AI service code, selected model artifact, CMUdict, content metadata/enriched CSVs, and the API contract.
