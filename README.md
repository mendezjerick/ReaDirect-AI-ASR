# ReaDirect AI/ASR

ReaDirect-AI-ASR is the separate Python research and service repository for ReaDirect speech and reading analysis work. The main Laravel ReaDirect application remains in its own repository. This repo handles dataset preparation, ASR experiments, pronunciation and reading-error detection, expected-answer comparison, and future model/service work that Laravel can call.

This repository is not the Laravel app, not a content-management dashboard, and not a place to commit real learner audio, identifiable learner metadata, private data, API keys, or model checkpoints.

## Main Pipeline

```text
learner audio
-> ASR transcription
-> transcript normalization
-> expected answer comparison
-> similarity/error type detection
-> output JSON
-> Laravel ReaDirect app uses result for scoring/feedback
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

Copy the example environment file before running local services:

```powershell
Copy-Item .env.example .env
```

Run the starter FastAPI service:

```powershell
uvicorn api.main:app --reload --port 8001
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

## Dataset Privacy Rules

- Do not commit child or learner audio.
- Do not commit identifiable learner metadata.
- Use anonymized learner IDs.
- Keep raw datasets local or in approved secure storage.
- Do not commit generated audio files.
- Do not commit model checkpoints or downloaded model weights.
- Do not commit `.env` or API keys.

## Dataset Manifest Format

The expected manifest is a CSV with these columns:

| Column | Purpose |
| --- | --- |
| `recording_id` | Stable recording identifier. |
| `learner_id_anonymized` | Anonymized learner identifier only. |
| `grade_level` | Learner grade level or band. |
| `prompt_id` | Content-bank or assessment prompt identifier. |
| `prompt_type` | Prompt type, such as letter, word, sentence, passage. |
| `module_key` | ReaDirect module key when applicable. |
| `activity_type` | Activity category. |
| `expected_text` | Primary expected answer. |
| `accepted_answers` | Alternate accepted answers, separated by `|` when stored in CSV. |
| `audio_path` | Relative path to local audio under `data/raw` or configured base path. |
| `duration_seconds` | Audio duration when known. |
| `manual_transcript` | Human transcript when available. |
| `asr_transcript` | ASR output when generated. |
| `human_correct` | Human correctness label when available. |
| `error_type` | Error type label when available. |
| `notes` | Non-identifying notes only. |

## Content Bank Connection

Safe CSVs from the main ReaDirect content bank can be copied into `content_bank/`. These files provide expected answers, accepted answers, item IDs, module tags, activity types, and difficulty metadata. This repo uses those content files to build dataset manifests and compare ASR output against expected answers.

Content-bank CSVs may be tracked only when they contain safe instructional content and no private learner data.

## Current Repository Context

This checkout currently includes `ReaDirect-Dataset/`, which appears to be a content-bank export with assessment, module, agent, rule, and feedback CSVs. Phase AI-1 leaves that folder unchanged and creates a separate Python ASR foundation around it.

## AI Phase 2 Dataset Bridge

Phase 2 connects ReaDirect content CSVs, CMUdict phoneme mappings, and future audio/transcript datasets through a unified content index and dataset manifest.

Active dataset plan:

- ReaDirect CSV content bank.
- CMUdict in `external_datasets/cmudict/`.
- Speechocean762 in `external_datasets/speechocean762/`.

Future optional / research-only datasets:

- L2-ARCTIC is removed from the active plan because it is CC BY-NC 4.0 / non-commercial. It must not be used for deployable model training, production inference, or commercial/government-client evaluation unless a compatible license is obtained.
- PF-STAR is removed from the active plan because it requires access/request. It should only be considered later as optional.

Expected CMUdict files:

```text
external_datasets/cmudict/cmudict.dict
external_datasets/cmudict/cmudict.phones
external_datasets/cmudict/cmudict.symbols
```

Inspect the current content bank:

```powershell
python scripts/inspect_content_bank.py --content-bank content_bank --cmudict external_datasets/cmudict
```

Inspect the existing local export without copying it:

```powershell
python scripts/inspect_content_bank.py --content-bank ReaDirect-Dataset --cmudict external_datasets/cmudict
```

Import a content-bank ZIP exported from the Laravel repo:

```powershell
python scripts/import_content_bank_zip.py --zip-path path/to/readirect-content-bank-export.zip
python scripts/import_content_bank_zip.py --zip-path path/to/readirect-content-bank-export.zip --overwrite
```

Build an enriched content index:

```powershell
python scripts/build_content_index.py --content-bank content_bank --cmudict-dir external_datasets/cmudict --output data/manifests/content_index.csv
```

Build a dataset manifest from fake-template metadata:

```powershell
python scripts/build_manifest.py --metadata-csv data/manifests/metadata_template.csv --content-bank content_bank --content-index data/manifests/content_index.csv --audio-dir data/raw --output data/manifests/dataset_manifest.csv
```

Validate a dataset manifest:

```powershell
python scripts/validate_manifest.py --manifest data/manifests/dataset_manifest.csv --content-bank content_bank --audio-base data/raw
```

The manifest joins each recording to a `prompt_id`, expected answer, accepted answers, module/activity metadata, and CMUdict-derived phoneme fields. It is the bridge between audio recordings, manual transcripts, ASR transcripts, and later scoring/error analysis.

Generated manifests and content indexes under `data/manifests/` are ignored by Git, except for the safe template `data/manifests/metadata_template.csv`.

## AI Phase 3 Speechocean762

Speechocean762 is now the active public speech/pronunciation dataset for ReaDirect AI/ASR experiments. Place the downloaded archive here:

```text
external_datasets/speechocean762/raw/speechocean762.tar.gz
```

Inspect the archive/extracted folder:

```powershell
python scripts/inspect_speechocean762.py --dataset-dir external_datasets/speechocean762 --print-tree
```

Safely extract:

```powershell
python scripts/extract_speechocean762.py --archive external_datasets/speechocean762/raw/speechocean762.tar.gz --dest external_datasets/speechocean762/extracted
```

Build the Speechocean762 manifest:

```powershell
python scripts/build_speechocean762_manifest.py --dataset-dir external_datasets/speechocean762/extracted --cmudict-dir external_datasets/cmudict --output data/manifests/speechocean762_manifest.csv
```

Validate it:

```powershell
python scripts/validate_manifest.py --manifest data/manifests/speechocean762_manifest.csv
```

Build the active public manifest:

```powershell
python scripts/build_public_dataset_manifest.py --speechocean-manifest data/manifests/speechocean762_manifest.csv --output data/manifests/unified_public_dataset_manifest.csv
```

Generate a readiness report:

```powershell
python scripts/report_dataset_readiness.py --manifest data/manifests/speechocean762_manifest.csv --output reports/speechocean762_readiness.md
```

Speechocean762 manifest rows preserve sentence scores, word scores, phoneme scores, word labels, phoneme labels, speaker age/gender metadata, split, transcript text, audio path, duration, and CMUdict-derived expected phonemes where available.

## AI Phase 4 ASR Baseline

Phase 4 measures pretrained ASR performance on Speechocean762 before any fine-tuning. The goal is to decide whether pretrained faster-whisper output is good enough for ReaDirect expected-answer comparison and pronunciation/error analysis.

No model is trained or fine-tuned in this phase.

Recommended first runs:

- `base.en` on CPU with `int8` compute for quick testing.
- `small.en` only if the machine can handle the extra runtime and memory.

Install dependencies:

```powershell
pip install -r requirements.txt
```

`faster-whisper` can take time to install and may download model files into the local model cache when first used. Do not commit model cache files.

Quick sample:

```powershell
python scripts/run_phase4_sample.py --limit 5 --model-size base.en --device cpu --compute-type int8
```

Limited baseline:

```powershell
python scripts/run_asr_baseline.py --manifest data/manifests/speechocean762_manifest.csv --output data/manifests/speechocean762_asr_baseline.csv --model-size base.en --device cpu --compute-type int8 --limit 50
```

Evaluate:

```powershell
python scripts/evaluate_asr_baseline.py --input data/manifests/speechocean762_asr_baseline.csv --output reports/asr_baseline_summary.md --metrics-csv reports/asr_baseline_metrics.csv
```

Interpretation:

- WER measures word-level transcription error.
- CER measures character-level transcription error, useful for short words.
- Exact match rate approximates whether ASR output can support direct expected-answer comparison.
- Short-word summaries matter for ReaDirect items such as `cat`, `dog`, `sun`, `pen`, `map`, `cup`, `hat`, `pig`, `run`, and `box`.

Fine-tuning is likely needed later if WER/CER are high on short words, exact match is low for simple utterances, or common substitutions interfere with expected-answer scoring.

## AI Phase 5 Reading Analysis

Phase 5 converts an ASR transcript into ReaDirect reading-analysis signals. It is heuristic and explainable. It does not train or fine-tune a model, and it does not make Laravel's official score decision. The Laravel app remains responsible for official rule-based scoring.

The AI repo provides analysis signals:

- `similarity_label`
- `error_type`
- `feedback_hint`
- `coach_hint_key`
- `skill_signal`
- transcript-derived phoneme comparison
- recommended practice focus

Analyze ASR output:

```powershell
python scripts/analyze_asr_outputs.py --input data/manifests/speechocean762_asr_baseline.csv --output data/manifests/speechocean762_reading_analysis.csv --cmudict-dir external_datasets/cmudict
```

Generate a report:

```powershell
python scripts/report_reading_analysis.py --input data/manifests/speechocean762_reading_analysis.csv --output reports/reading_analysis_summary.md
```

Run API:

```powershell
uvicorn api.main:app --reload --port 8001
```

Example `/analyze-text` request:

```json
{
  "expected_text": "cat",
  "actual_text": "cap",
  "accepted_answers": ["cat"]
}
```

Example result signals:

```text
similarity_label = very_close
error_type = final_sound_error
skill_signal = final_consonant
feedback_hint = ending_sound
```

Limitations:

- Actual phonemes are derived from the ASR transcript, not direct acoustic phoneme recognition.
- ASR mistakes can affect error detection.
- This is not yet a trained pronunciation model.
- Speechocean annotations may later improve scoring and validation.

## AI Phase 6 Content Enrichment

Phase 6 enriches ReaDirect content-bank rows with phoneme tags, skill tags, target phonemes, difficulty metadata, and adaptive selection metadata. This makes module items easier to use for targeted practice, remediation, review, and mastery checks.

Generated enriched files are ignored by Git by default. Review them before importing anything back into the main Laravel ReaDirect repository.

Run enrichment:

```powershell
python scripts/enrich_content_bank.py --content-bank content_bank --cmudict-dir external_datasets/cmudict --output-dir content_bank_enriched
```

If `content_bank/` is empty but the Phase 2 index exists, use:

```powershell
python scripts/enrich_content_bank.py --content-bank content_bank --content-index data/manifests/content_index.csv --cmudict-dir external_datasets/cmudict --output-dir content_bank_enriched --write-import-ready
```

Validate:

```powershell
python scripts/validate_enriched_content.py --input content_bank_enriched/enriched_content_index.csv
```

Generate report:

```powershell
python scripts/report_content_enrichment.py --enriched-index content_bank_enriched/enriched_content_index.csv --output content_bank_enriched/reports/content_enrichment_report.md
```

Export review ZIP:

```powershell
python scripts/export_enriched_content_zip.py --source-dir content_bank_enriched/import_ready --output content_bank_enriched/readirect-enriched-content.zip
```

API content-item preview:

```powershell
uvicorn api.main:app --reload --port 8001
```

POST `/analyze-content-item`:

```json
{
  "prompt_id": "M2-001",
  "expected_text": "cat",
  "activity_type": "read_word",
  "module_key": "module_2"
}
```

Limitations:

- CMUdict is an American English pronunciation dictionary.
- Some Grade 1 words, names, or local terms may be missing.
- Phoneme tags are dictionary-derived, not acoustic.
- Difficulty scoring is heuristic and should be reviewed by educators.

## AI Phase 7 FastAPI Service

Phase 7 exposes the Laravel-facing AI analysis service. Laravel sends expected-answer context and either text or an audio path. The AI service returns analysis signals only; Laravel remains the official scorer.

Run locally:

```powershell
uvicorn api.main:app --reload --port 8001
```

Health:

```powershell
curl http://127.0.0.1:8001/health
```

Analyze text:

```powershell
python scripts/test_api_analysis.py --mode text --expected-text cat --actual-text cap --accepted-answer cat --debug
```

Analyze audio path:

```powershell
python scripts/test_api_analysis.py --mode audio --audio-path data/samples/sample.wav --expected-text cat --accepted-answer cat --debug
```

Local defaults use `ASR_PROVIDER=mock`, so collaborators can run the API without downloading Whisper. Set `ASR_PROVIDER=faster_whisper` only when ready for real transcription.

API token protection is optional:

```text
API_AUTH_ENABLED=true
READIRECT_AI_API_TOKEN=your-server-token
```

When enabled, Laravel must send:

```text
X-ReaDirect-AI-Token: your-server-token
```

CORS defaults allow local Laravel development at `http://127.0.0.1:8000` and `http://localhost:8000`. In production, keep the AI service private and have Laravel call it server-to-server.

## AI Phase 8 Adaptive Tutoring Engine

Phase 8 adds a heuristic recommendation layer that suggests the next practice item or learning action from learner history plus enriched content metadata. It uses `error_type`, `skill_signal`, phoneme tags, difficulty, module context, mastery/review flags, and recent attempts.

The engine is advisory only. Laravel remains responsible for official scoring, module progression, persistence, and deciding whether to accept or override the recommendation.

The adaptive engine can use candidates from:

- `content_bank_enriched/enriched_content_index.csv` when available.
- `data/manifests/content_index.csv` as fallback.
- Request-provided `candidate_items` from Laravel as a final fallback.

Recommend the next item:

```json
POST http://127.0.0.1:8001/recommend-next
{
  "learner_history": [
    {
      "prompt_id": "M2-001",
      "expected_text": "cat",
      "actual_text": "cap",
      "is_correct": false,
      "error_type": "final_sound_error",
      "skill_signal": "final_consonant",
      "target_phoneme": "T",
      "difficulty_level": "easy"
    }
  ],
  "candidate_items": [
    {
      "prompt_id": "M2-014",
      "module_key": "module_2",
      "activity_type": "read_word",
      "prompt_text": "Read the word.",
      "expected_text": "hat",
      "error_focus": "final_consonant",
      "target_phoneme": "T",
      "difficulty_level": "easy",
      "is_active": true,
      "needs_manual_review": false
    }
  ],
  "top_k": 5,
  "debug": true
}
```

Simulate recommendations without Laravel:

```powershell
python scripts/simulate_adaptive_tutoring.py --top-k 5
```

`/analyze-text` and `/analyze-audio` still work without learner history. If `learner_history` is included, the response also includes `adaptive_recommendation` and `learner_summary`.

See `docs/ADAPTIVE_TUTORING_ENGINE.md` for the scoring policy, schema, examples, and limitations.

## AI Phase 9 Fine-Tuning Decision Workflow

Phase 9 does not fine-tune Whisper. It decides whether fine-tuning is justified from baseline ASR evidence and prepares Whisper-compatible JSONL datasets only if needed.

The decision uses WER, CER, exact match rate, ReaDirect short-word accuracy, blank ASR output rate, and dataset readiness checks for row count, duration, transcript coverage, and audio availability.

Decide whether fine-tuning is justified:

```powershell
python scripts/decide_finetuning.py --manifest data/manifests/speechocean762_manifest.csv --baseline data/manifests/speechocean762_asr_baseline.csv --output reports/finetuning_decision.md
```

Prepare a Whisper-compatible dataset after the decision supports it:

```powershell
python scripts/prepare_whisper_finetune_dataset.py --manifest data/manifests/speechocean762_manifest.csv --output-dir data/processed/whisper_finetune
```

Preview the future training config without training:

```powershell
python training/train_whisper_skeleton.py --config configs/whisper_finetune_config.yaml
```

Collaborators do not need to retrain by default. One approved training run can later produce a reviewed model artifact outside Git, stored locally under `model_artifacts/` and shared through secure external storage.

See `docs/FINETUNING_DECISION_WORKFLOW.md` and `docs/WHISPER_FINETUNING_PLAN.md`.

## Planned Phases

1. AI Phase 1: Repo setup and dataset format.
2. AI Phase 2: Content bank import and manifest builder.
3. AI Phase 3: Speechocean762 loader and public manifest conversion.
4. AI Phase 4: Baseline ASR using faster-whisper on Speechocean762, WER/CER, exact match, and pronunciation-relevant summaries.
5. AI Phase 5: ReaDirect-specific word similarity, expected-answer comparison, and pronunciation error detection using ASR outputs + CMUdict tags.
6. AI Phase 6: Enrich ReaDirect CSV content bank with phoneme tags, skill tags, target phonemes, and adaptive item-selection metadata.
7. AI Phase 7: FastAPI analysis service for Laravel integration.
8. AI Phase 8: Adaptive tutoring and next-item recommendation.
9. AI Phase 9: Fine-tuning decision workflow and dataset preparation.
10. AI Phase 10: Optional guarded Whisper fine-tuning implementation if Phase 9 recommends it.

## Useful Commands

Inspect a manifest:

```powershell
python scripts/inspect_dataset.py --manifest data/manifests/dataset_manifest.csv
```

Build a starter manifest from local audio:

```powershell
python scripts/build_manifest.py --audio-dir data/raw --output data/manifests/dataset_manifest.csv
```

Validate a manifest:

```powershell
python scripts/validate_manifest.py --manifest data/manifests/dataset_manifest.csv
```
