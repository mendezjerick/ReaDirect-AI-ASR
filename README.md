# ReaDirect-AI-ASR

ReaDirect-AI-ASR serves as the artificial intelligence layer of the ReaDirect system. It uses automatic speech recognition, fine-tuned Whisper-based transcription, phoneme-aware answer analysis, expected-answer comparison, pronunciation-error detection, and adaptive tutoring logic to interpret learner oral reading responses. Instead of functioning as a simple quiz checker, the AI component processes learner speech, compares it with expected reading targets, identifies similarity and possible error types, and provides structured signals that the main ReaDirect system can use for feedback, intervention, and adaptive activity selection.

This repository is separate from the main ReaDirect Laravel application. The Laravel repository remains responsible for the application interface, official scoring, database storage, authentication, and learner workflow. This Python repository provides the research, model, analysis, and FastAPI service layer that Laravel can call.

## 1. Data Preparation

Data preparation organizes public speech datasets, ReaDirect content files, manifests, phoneme resources, model-ready training files, reports, and local model artifacts. This structure allows the AI pipeline to connect each audio recording with a transcript, expected answer, prompt ID, phoneme metadata, ASR output, and evaluation result.

Important folders:

- `external_datasets/`: local public datasets and CMUdict resources.
- `data/raw/`: local raw audio or dataset files, ignored by Git.
- `data/processed/`: generated processed files, including Whisper fine-tuning JSONL files.
- `data/manifests/`: generated dataset manifests and content indexes.
- `content_bank/`: safe ReaDirect CSV content-bank imports.
- `content_bank_enriched/`: generated phoneme/adaptive metadata exports.
- `model_artifacts/`: local fine-tuned model outputs and converted model folders.
- `reports/`: generated evaluation and readiness reports.

Large datasets, generated manifests, model checkpoints, training logs, private data, and report artifacts are intentionally ignored by Git.

## 2. Data Acquisition & Collection

### Speechocean762

Speechocean762 is the primary public speech/pronunciation dataset used in this repository for ASR baseline evaluation and fine-tuning experiments. It contains English pronunciation-assessment utterances and is stored locally under:

```text
external_datasets/speechocean762/
```

Expected local paths:

```text
external_datasets/speechocean762/raw/speechocean762.tar.gz
external_datasets/speechocean762/extracted/
```

The archive and extracted audio are not committed to GitHub because they are large dataset files.

### CMUdict

CMUdict is used as the pronunciation dictionary. It provides word-to-phoneme mappings that support phoneme-aware comparison and content enrichment.

Expected files:

```text
external_datasets/cmudict/cmudict.dict
external_datasets/cmudict/cmudict.phones
external_datasets/cmudict/cmudict.symbols
```

### ReaDirect Curated CSV Content Bank

The ReaDirect content bank contains expected reading items, module activities, assessment items, accepted answers, and learning content. These CSVs are imported from the main ReaDirect content bank and are used to create:

```text
data/manifests/content_index.csv
content_bank_enriched/
```

These CSVs represent the target reading content that the AI analysis compares against learner speech.

### Removed or Excluded Datasets

- L2-ARCTIC was removed from the active plan because of non-commercial licensing concerns.
- PF-STAR was removed from the active plan because it requires access/request approval.

Neither dataset is part of the deployable or commercial-facing workflow.

## 3. Data Cleaning & Preprocessing

The Speechocean762 archive was inspected, extracted, and converted into a unified manifest. Audio paths were validated, transcript availability was checked, durations were included, and rows were prepared for train/validation/test splits.

Relevant scripts:

- `scripts/inspect_speechocean762.py`
- `scripts/extract_speechocean762.py`
- `scripts/build_speechocean762_manifest.py`
- `scripts/validate_manifest.py`
- `scripts/prepare_whisper_finetune_dataset.py`

Generated outputs:

```text
data/manifests/speechocean762_manifest.csv
data/processed/whisper_finetune/train.jsonl
data/processed/whisper_finetune/validation.jsonl
data/processed/whisper_finetune/test.jsonl
```

These files are generated locally when reproducing the workflow and are ignored by Git.

## 4. AI/ASR Methodology

The system follows this AI pipeline:

```text
learner audio
-> ASR transcription
-> transcript normalization
-> expected-answer comparison
-> CMUdict phoneme mapping
-> phoneme-aware comparison
-> error type detection
-> feedback hint generation
-> adaptive recommendation
```

The architecture is hybrid:

- Faster-Whisper / Whisper performs speech recognition.
- Hugging Face Transformers supports Whisper fine-tuning.
- CMUdict provides pronunciation and phoneme information.
- Explainable heuristic rules detect reading and pronunciation-related error types.
- FastAPI exposes the analysis service to the Laravel application.

Laravel remains the official scorer and progression controller. The AI service provides analysis signals that Laravel may use for feedback, intervention, and adaptive practice.

## 5. Baseline ASR Evaluation

A pretrained Whisper/faster-whisper model was evaluated before fine-tuning to determine whether fine-tuning was justified.

Evaluation metrics:

- WER: word error rate. Lower is better.
- CER: character error rate. Lower is better.
- Exact match rate: percentage of normalized ASR outputs that exactly matched the reference transcript.
- Short-word accuracy where applicable.

Relevant scripts:

- `scripts/run_asr_baseline.py`
- `scripts/evaluate_asr_baseline.py`
- `scripts/report_dataset_readiness.py`

Baseline result:

| Metric | Baseline |
|---|---:|
| WER | 0.494 |
| CER | 0.324 |
| Exact Match Rate | 0.340 |

The baseline result showed that the pretrained model was useful but not accurate enough for the target reading-assessment workflow without additional analysis and improvement.

## 6. Whisper Fine-Tuning

The Phase 9 decision workflow recommended fine-tuning with high confidence. Fine-tuning was performed using Hugging Face Transformers.

Training setup:

- Model: `openai/whisper-base.en`
- GPU: NVIDIA GeForce RTX 3060
- VRAM: 12GB
- RAM: 32GB DDR4
- Training data: Speechocean762 JSONL files prepared from the unified manifest
- Output folder: `model_artifacts/readirect-whisper-base-en-v1-hf/`

Relevant scripts:

- `training/train_whisper.py`
- `scripts/check_training_environment.py`
- `scripts/prepare_whisper_finetune_dataset.py`
- `scripts/evaluate_finetuned_whisper.py`
- `scripts/convert_whisper_to_faster_whisper.py`

Training is manual and explicit. The repository contains the pipeline and configuration, but the actual model artifact is not committed to GitHub. The final model can be shared separately through private storage.

## 7. Baseline vs Fine-Tuned Model Comparison

| Metric | Baseline | Fine-Tuned | Interpretation |
|---|---:|---:|---|
| WER | 0.494 | 0.396 | Improved overall word transcription |
| CER | 0.324 | 0.197 | Improved character-level accuracy |
| Exact Match Rate | 0.340 | 0.304 | Slightly lower exact-match rate; strict exact matching remains challenging |

Fine-tuning improved WER from 0.494 to 0.396 and CER from 0.324 to 0.197. Exact match decreased slightly from 0.340 to 0.304, which means the fine-tuned model generally produces transcripts that are closer to the reference but still does not guarantee exact answer matching.

This result is important for ReaDirect: the system should not rely on exact ASR transcript matching alone. ReaDirect combines ASR with similarity scoring, phoneme-aware comparison, accepted answers, error-type detection, and fallback logic to make analysis more useful and fair.

The fine-tuned model is useful as an improved ASR layer, but it should be combined with the explainable reading-analysis engine rather than used as the sole basis for learner scoring.

## 8. ReaDirect Reading Analysis Engine

The reading analysis engine compares ASR transcripts against expected reading targets and returns structured analysis fields.

Outputs include:

- `is_correct`
- `similarity_label`
- `character_similarity`
- `token_similarity`
- `expected_phonemes`
- `actual_phonemes`
- `phoneme_similarity`
- `error_type`
- `feedback_hint`
- `skill_signal`
- `recommended_practice_focus`

Relevant modules:

- `src/readirect_asr/scoring/answer_matching.py`
- `src/readirect_asr/scoring/phoneme_comparison.py`
- `src/readirect_asr/scoring/error_detection.py`
- `src/readirect_asr/scoring/feedback_hints.py`
- `src/readirect_asr/scoring/skill_signals.py`
- `src/readirect_asr/scoring/reading_analyzer.py`

Example:

```text
Expected: cat
ASR transcript: cap
```

Result:

- `similarity_label`: `very_close`
- `error_type`: `final_sound_error`
- `skill_signal`: `final_consonant`
- `feedback_hint`: `listen_to_final_sound`

This is what allows the system to act more like an AI reading tutor instead of a simple correct/incorrect checker.

## 9. Content Bank Enrichment

The ReaDirect CSV content was enriched with phoneme and adaptive learning metadata. Because `content_bank/` may contain placeholders, enrichment can also use:

```text
data/manifests/content_index.csv
```

Generated enriched outputs can be created under:

```text
content_bank_enriched/
```

Enrichment fields include:

- `expected_phonemes`
- `initial_phoneme`
- `vowel_phonemes`
- `final_phoneme`
- `phoneme_pattern`
- `skill_tag`
- `error_focus`
- `target_phoneme`
- `difficulty_level`
- `adaptive_bucket`
- `recommended_for_error_type`
- `needs_manual_review`

Relevant scripts:

- `scripts/enrich_content_bank.py`
- `scripts/validate_enriched_content.py`
- `scripts/report_content_enrichment.py`
- `scripts/export_enriched_content_zip.py`

The enriched CSVs help the system recommend targeted activities based on learner weaknesses.

## 10. Adaptive Tutoring and Recommendation Engine

The adaptive engine uses learner history and AI analysis signals to recommend the next item or practice focus. It is advisory only; Laravel remains the official progression controller.

Inputs:

- `error_type`
- `skill_signal`
- `target_phoneme`
- `difficulty_level`
- recent learner history
- candidate items from Laravel or the content repository

Outputs:

- `selected_item`
- `ranked_candidates`
- `recommended_action`
- `difficulty_adjustment`
- `teacher_explanation`
- `learner_safe_summary`

Relevant modules:

- `src/readirect_asr/adaptive/learner_state.py`
- `src/readirect_asr/adaptive/remediation_policy.py`
- `src/readirect_asr/adaptive/difficulty_policy.py`
- `src/readirect_asr/adaptive/item_selector.py`
- `src/readirect_asr/adaptive/recommendation.py`
- `src/readirect_asr/adaptive/explanation.py`

## 11. FastAPI Service for Laravel Integration

This repository exposes a FastAPI service that the main ReaDirect Laravel app can call. Students do not call this AI service directly. Laravel sends an audio path, expected answer, accepted answers, prompt ID, and relevant context. The AI service returns structured analysis signals.

Endpoints:

- `GET /health`
- `GET /version`
- `POST /analyze-text`
- `POST /analyze-audio`
- `POST /content-item`
- `POST /recommend-next`

Integration documents:

- `docs/FASTAPI_SERVICE.md`
- `docs/LARAVEL_INTEGRATION_CONTRACT.md`

Example flow:

```text
Laravel
-> sends audio + expected answer
-> AI service transcribes/analyzes
-> returns transcript, error_type, similarity, skill_signal
-> Laravel saves result and generates feedback
```

## 12. Repository Structure

```text
ReaDirect-AI-ASR/
├── api/
├── configs/
├── src/readirect_asr/
├── scripts/
├── training/
├── docs/
├── tests/
├── external_datasets/
├── data/
├── content_bank/
├── content_bank_enriched/
├── model_artifacts/
└── reports/
```

Folder summary:

- `api/`: FastAPI endpoints and service orchestration.
- `configs/`: ASR, service, dataset, adaptive, and fine-tuning configs.
- `src/readirect_asr/`: core Python package for ASR, phonemes, scoring, content, evaluation, adaptive logic, and fine-tuning.
- `scripts/`: command-line tools for dataset preparation, evaluation, analysis, and reporting.
- `training/`: guarded Whisper fine-tuning scripts.
- `docs/`: supporting technical documentation.
- `tests/`: automated tests.
- `external_datasets/`: local public datasets and CMUdict.
- `data/`: local raw, processed, and manifest artifacts.
- `content_bank/`: ReaDirect content-bank imports.
- `content_bank_enriched/`: generated enriched content outputs.
- `model_artifacts/`: local fine-tuned or converted model artifacts.
- `reports/`: generated evaluation and process reports.

## 13. What Is Included in GitHub

The repository includes:

- source code
- FastAPI service
- configuration files
- dataset and evaluation scripts
- fine-tuning pipeline code
- tests
- documentation
- CMUdict files if present and license-appropriate
- placeholder folders such as `.gitkeep`
- safe example templates

## 14. What Is Not Included in GitHub

The repository intentionally excludes:

- Speechocean762 archive
- extracted Speechocean762 audio
- generated manifests
- generated reports
- fine-tuned model checkpoints
- model artifacts
- `.env`
- private learner data
- real learner audio

These files are excluded because of file size, licensing, privacy, data ownership, and GitHub hygiene.

## 15. How to Reproduce the Workflow

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check GPU:

```powershell
python scripts/check_training_environment.py
```

Extract Speechocean762:

```powershell
python scripts/extract_speechocean762.py --archive external_datasets/speechocean762/raw/speechocean762.tar.gz --dest external_datasets/speechocean762/extracted
```

Build manifest:

```powershell
python scripts/build_speechocean762_manifest.py --dataset-dir external_datasets/speechocean762/extracted --cmudict-dir external_datasets/cmudict --output data/manifests/speechocean762_manifest.csv
```

Run baseline ASR:

```powershell
python scripts/run_asr_baseline.py --manifest data/manifests/speechocean762_manifest.csv --output data/manifests/speechocean762_asr_baseline.csv --model-size base.en --device cuda --compute-type float16
```

Evaluate baseline:

```powershell
python scripts/evaluate_asr_baseline.py --input data/manifests/speechocean762_asr_baseline.csv --output reports/asr_baseline_summary.md --metrics-csv reports/asr_baseline_metrics.csv
```

Prepare fine-tuning dataset:

```powershell
python scripts/prepare_whisper_finetune_dataset.py --manifest data/manifests/speechocean762_manifest.csv --output-dir data/processed/whisper_finetune
```

Fine-tune:

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --run
```

Evaluate fine-tuned model:

```powershell
python scripts/evaluate_finetuned_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --test-jsonl data/processed/whisper_finetune/test.jsonl --output reports/finetuned_whisper_eval.md --metrics-json reports/finetuned_whisper_metrics.json
```

Run FastAPI:

```powershell
uvicorn api.main:app --reload --port 8001
```

Run tests:

```powershell
pytest
```

## 16. Current Results and Interpretation

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| WER | 0.494 | 0.396 |
| CER | 0.324 | 0.197 |
| Exact Match Rate | 0.340 | 0.304 |

Fine-tuning improved WER and CER, which means the fine-tuned model generally transcribes closer to the reference than the pretrained baseline. Exact match remains challenging, especially for short reading responses and strict answer matching.

For this reason, ReaDirect uses a hybrid AI architecture instead of relying only on exact transcription. ASR output is combined with expected-answer matching, accepted-answer handling, phoneme-aware comparison, feedback hints, error types, and adaptive tutoring signals.

## 17. Limitations

- The fine-tuned model was trained and evaluated on Speechocean762, not actual ReaDirect learner recordings.
- ASR may still mishear short words.
- Phoneme comparison is transcript-derived, not direct acoustic phoneme recognition.
- Exact match is not reliable enough as the only scoring mechanism.
- Teacher/admin review and Laravel rule-based scoring remain important.
- Further improvements may require consented real learner recordings, better preprocessing, or more training data.

## 18. Next Steps

- Import reviewed enriched CSVs into the main Laravel repository.
- Integrate the FastAPI service into the main ReaDirect Laravel app.
- Add a Laravel `AIAnalysisService` client.
- Store transcript, `error_type`, similarity, and skill-signal fields in the database.
- Add an admin debug view for AI results.
- Use adaptive recommendations with Laravel eligibility rules.
- Optionally convert the fine-tuned model to faster-whisper/CTranslate2 format.
- Evaluate the model on real pilot data later with consent.
