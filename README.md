# ReaDirect-AI-ASR

ReaDirect-AI-ASR is the Python/FastAPI AI speech service for the ReaDirect reading system. It is separate from the main ReaDirect Laravel application: Laravel owns the learner workflow, authentication, persistence, official scoring rules, and UI, while this repository provides local ASR, transcript correction, pronunciation evidence, audio quality validation, and analysis metadata.

The current runtime is Wav2Vec2-only. Whisper was previously used or referenced in older experiments and documents, but it has been removed from the active runtime.

## 1. Project Overview

The service accepts learner audio and expected reading text from Laravel/content CSVs, runs local Wav2Vec2 ASR, preserves the raw transcript, applies expected-centric correction where appropriate, and returns structured scoring metadata for Laravel to use.

Core responsibilities:

- Local Wav2Vec2 speech recognition.
- Wav2Vec2 phoneme evidence for letter and short-word support.
- Expected-centric letter and word correction.
- Separate `raw_transcript`, `corrected_transcript`, and `displayed_transcript` handling.
- Human-curated reinforcement correction memory.
- Audio quality validation and uncertainty metadata.
- Sentence scoring support metadata without forcing full sentences to the expected text.

## 2. Current ASR Runtime

Active runtime:

- Architecture: Wav2Vec2-only.
- Active ASR model: `models/wav2vec2-readirect-asr-letters-v2`.
- Reference/fallback ASR model: `models/wav2vec2-readirect-asr`.
- Phoneme support model: `models/wav2vec2-phoneme`.
- Decoding: CTC greedy decoding.
- Whisper status: removed from runtime and deprecated for current implementation.

The ASR service does not use Whisper as the active model, sentence model, fallback model, or current route. Any old Whisper references are historical and should not be followed for current runtime work.

## 3. Repository Structure

```text
ReaDirect-AI-ASR/
|-- api/
|-- src/readirect_asr/
|-- models/
|-- external_datasets/
|-- reinforcement-learning/
|-- configs/
|-- scripts/
|-- docs/
|-- outputs/
|-- checkpoints/
|-- tests/
|-- training/
|-- reports/
|-- content_bank/
|-- content_bank_enriched/
`-- data/
```

Folder summary:

- `api/`: FastAPI app, routes, request/response handling, health/status endpoints, and service orchestration.
- `src/readirect_asr/`: core ASR, phoneme, scoring, correction, audio validation, content, evaluation, and adaptive analysis package.
- `models/`: local Hugging Face model folders used by runtime, including `wav2vec2-readirect-asr-letters-v2`, `wav2vec2-readirect-asr`, and `wav2vec2-phoneme`.
- `external_datasets/`: local public and custom datasets prepared for training, evaluation, and analysis.
- `reinforcement-learning/`: human-curated correction memory tables; this is not automatic reinforcement learning or model training.
- `configs/`: service and training configuration references.
- `scripts/`: dataset preparation, validation, training, evaluation, reporting, startup, and integration utilities.
- `docs/`: supporting technical documentation.
- `outputs/`: generated evaluation/model-comparison outputs and interpretation reports.
- `checkpoints/`: local training checkpoints; not part of runtime unless explicitly promoted after validation.
- `tests/`: automated tests.
- `training/`: shared training helpers and historical training scripts.
- `reports/`: generated dataset, cleaning, evaluation, and readiness reports.
- `content_bank/`: ReaDirect CSV content-bank imports.
- `content_bank_enriched/`: generated content metadata with phoneme/adaptive fields.
- `data/`: local raw, processed, and manifest artifacts.

## 4. ASR Processing Pipeline

Current runtime flow:

```text
Audio input
-> audio loading / mono conversion / 16 kHz resampling
-> audio quality validation
-> Wav2Vec2 ASR
-> CTC greedy decoding
-> raw_transcript
-> Wav2Vec2 phoneme evidence
-> expected-centric correction
-> corrected_transcript
-> displayed_transcript
-> scoring metadata
-> API response
```

For sentence prompts, the service returns transcript and scoring support metadata such as WER/CER, acceptance state, audio quality, pause metrics, and debug metadata. It does not force the full displayed sentence to `expected_text`.

## 5. Transcript Correction System

The correction layer is expected-centric: it uses the expected answer from Laravel/content CSVs as the target and decides whether a raw Wav2Vec2 recognition should be accepted or corrected for scoring/display.

Transcript fields:

- `raw_transcript`: direct Wav2Vec2 ASR output.
- `wav2vec2_transcript`: Wav2Vec2 transcript field retained for explicit model provenance.
- `corrected_transcript`: transcript used for scoring when correction is accepted.
- `displayed_transcript`: transcript shown to the learner.
- `expected_text`: target answer from Laravel/content CSV.
- `transcript`: backward-compatible learner-facing transcript, generally aligned with `displayed_transcript`.

Letter and word prompts:

- Accepted letter/word corrections set `corrected_transcript = expected_text`.
- Accepted letter/word corrections set `displayed_transcript = expected_text`.
- Rejected answers keep the recognized output for display.
- Correction can use raw transcript similarity, phoneme evidence, critical phoneme rules, edit distance, and reinforcement memory matches.

Sentence prompts:

- Full displayed sentences are not forced to `expected_text`.
- Sentence scoring uses the Wav2Vec2 transcript plus support metadata such as WER, CER, exact match, pause metrics, and uncertainty flags.

Critical phoneme rules protect against over-correction. A letter or word correction should only be accepted when the observed phoneme evidence is compatible with the expected target and does not contradict critical target sounds.

## 6. Reinforcement Correction Memory

The `reinforcement-learning/` folder stores a human-curated correction table for known ASR error patterns. It is not model training, not automatic self-learning, and not true reinforcement learning in the ML sense.

Example:

```text
expected_text = Z
raw_transcript = They
correction result = accepted
corrected_transcript = Z
displayed_transcript = Z
```

The correction only applies when the expected target and raw transcript match a curated rule. If the expected target is different, the same raw transcript is not blindly corrected.

Relevant documentation:

- `docs/REINFORCEMENT_CORRECTIONS.md`

## 7. Audio Quality and Uncertainty Handling

The runtime performs lightweight audio validation before and around ASR. This validation produces metadata and can request a retry; it does not train or modify any model.

Quality checks include:

- Duration validation.
- Silent audio detection.
- Low-volume detection.
- Clipping detection.
- Pause metadata.
- RMS, dBFS, silence ratio, clipping ratio, and related audio features where available.

Runtime uncertainty fields include:

- `retry_required`: the audio should be recorded again rather than scored as a normal wrong answer.
- `uncertain`: the transcript/correction should be treated cautiously.
- `audio_quality`: detailed quality metadata.
- `pause_metrics`: pause/silence metadata.

When `retry_required=true`, the service should avoid forcing `expected_text` into `displayed_transcript`.

Relevant documentation:

- `docs/AUDIO_QUALITY_VALIDATION.md`

## 8. Dataset Inventory

Datasets are stored locally under `external_datasets/`. Large dataset files, generated manifests, reports, and model artifacts are not expected to be committed to Git.

### LibriSpeech

Location:

```text
external_datasets/librispeech/
```

Purpose:

- General English ASR stability.
- Backbone/general read-speech data.
- Helps preserve broad English transcription behavior during Wav2Vec2 fine-tuning.

### SpeechOcean

Location:

```text
external_datasets/speechocean/
```

Purpose:

- Pronunciation-oriented speech data.
- Learner/child-relevant speech support.
- Pronunciation evaluation support.
- Helps preserve pronunciation-oriented behavior during continued fine-tuning.

Some older scripts/docs may refer to `speechocean762`; those references are historical naming details and should be reconciled carefully before new dataset work.

### Custom ReaDirect Letter Dataset

Location:

```text
external_datasets/readirect_letters/
```

Purpose:

- Isolated A-Z letter recognition.
- Hard-case letter evaluation.
- Correction validation.
- Planned v2 continued fine-tuning.

Expected structure:

```text
external_datasets/readirect_letters/
|-- processed_wav/
|   |-- speaker_001/
|   |-- speaker_002/
|   |-- speaker_003/
|   |-- speaker_004/
|   `-- speaker_005/
|-- manifests/
|   |-- readirect_letters_all.csv
|   |-- readirect_letters_all.jsonl
|   |-- readirect_letters_train.csv
|   |-- readirect_letters_train.jsonl
|   |-- readirect_letters_valid.csv
|   |-- readirect_letters_valid.jsonl
|   |-- readirect_letters_test.csv
|   `-- readirect_letters_test.jsonl
`-- reports/
    |-- DATA_CLEANING_REPORT.md
    |-- audio_quality_report.csv
    |-- folder_validation_report.csv
    |-- filename_validation_report.csv
    |-- skipped_files_report.csv
    `-- summary.json
```

Custom letter dataset summary:

| Metric | Count |
|---|---:|
| Total usable rows | 1300 |
| OK files | 1228 |
| Warning files | 72 |
| Critical files | 0 |
| Train rows | 780 |
| Valid rows | 260 |
| Test rows | 260 |
| Adult male files | 780 |
| Adult female files | 520 |

Speaker metadata:

| Folder / ID | Voice group |
|---|---|
| `speaker_001` / `spk001` | adult_male |
| `speaker_002` / `spk002` | adult_male |
| `speaker_003` / `spk003` | adult_male |
| `speaker_004` / `spk004` | adult_female |
| `speaker_005` / `spk005` | adult_female |

Important limitation: the custom letter dataset contains adult speaker recordings. It is useful for letter-level ASR adaptation, hard-case evaluation, and correction validation, but it does not fully represent child learner voices.

## 9. Analysis & Modeling

### A. Exploratory Data Analysis (EDA)

EDA in this repository means checking whether datasets, transcripts, audio, and model outputs are suitable for ReaDirect ASR and correction workflows.

EDA includes:

- Dataset counts.
- Speaker distribution.
- `voice_group` distribution.
- Letter distribution.
- Duration distribution.
- Warning and critical audio counts.
- ASR transcript errors.
- WER/CER and correction results.
- Hard-case letters and words.

EDA outputs may be found in dataset `reports/` folders and in `outputs/evaluation/`, including model comparison reports.

### B. Feature Engineering

Feature engineering in this repository prepares audio, text, phoneme evidence, and scoring metadata for ASR, correction, and evaluation.

Feature examples:

- Audio resampling to 16 kHz mono.
- Waveform preparation for Wav2Vec2.
- Text normalization.
- Expected phoneme generation.
- Observed phoneme extraction.
- Phoneme similarity.
- Levenshtein/edit-distance features.
- Audio quality features such as RMS, dBFS, silence ratio, and clipping ratio.
- Pause metrics.
- Transcript correction flags.
- Reinforcement correction matches.

### C. Model Development (ML / AI)

The active ASR model is a local Wav2Vec2ForCTC model stored at:

```text
models/wav2vec2-readirect-asr-letters-v2
```

The previous v1 ASR model remains available for fallback/reference at:

```text
models/wav2vec2-readirect-asr
```

The separate phoneme support model is stored at:

```text
models/wav2vec2-phoneme
```

The phoneme model remains separate and should not be fine-tuned unless safe phoneme-level labels exist.

Current active model:

- `models/wav2vec2-readirect-asr-letters-v2`

Previous v1 model retained:

- `models/wav2vec2-readirect-asr`

The v2 model is active for FastAPI ASR runtime. The v1 model is not deleted or overwritten.

V2 training strategy:

- 50% custom ReaDirect letter dataset.
- 30% SpeechOcean.
- 20% LibriSpeech.

Purpose:

- Improve isolated letter recognition.
- Preserve pronunciation-oriented behavior.
- Preserve general English ASR stability.
- Reduce overfitting to the custom adult letter recordings.

Do not overwrite v1 when preparing later models. Train any future model into a separate model folder and explicitly promote it through runtime configuration.

### D. Model Evaluation & Validation

Evaluation should happen before switching any future runtime model.

Metrics and checks include:

- WER.
- CER.
- Exact match.
- Accepted rate.
- Corrected WER/CER.
- Letter-level evaluation.
- Word-level evaluation.
- Sentence-level evaluation.
- Hard-case evaluation.
- Model comparison against the active or reference model.

The v2 model has been promoted to runtime. Future candidate models must not become active until they are evaluated against the current active model and explicitly approved.

## 10. Training and Fine-Tuning Workflow

Training is manual. Running the API service does not train a model, update weights, or switch runtime models.

Current Wav2Vec2 training scripts include:

- `scripts/prepare_librispeech_manifest.py`
- `scripts/prepare_speechocean_manifest.py`
- `scripts/build_wav2vec2_training_manifest.py`
- `scripts/validate_training_manifest.py`
- `scripts/train_wav2vec2_readirect_asr.py`
- `scripts/evaluate_wav2vec2_readirect_asr.py`
- `scripts/evaluate_model_comparison.py`

Existing v1-style manifest preparation commands:

```powershell
python scripts/prepare_librispeech_manifest.py
python scripts/prepare_speechocean_manifest.py
python scripts/build_wav2vec2_training_manifest.py
python scripts/validate_training_manifest.py external_datasets/manifests/readirect_train_mixed.jsonl
```

Existing v1-style guarded training command:

```powershell
python scripts/train_wav2vec2_readirect_asr.py --config configs/wav2vec2_readirect_asr.yaml --no-eval
```

Evaluation during training is disabled by default in `configs/wav2vec2_readirect_asr.yaml`. Post-training evaluation should be run explicitly and should compare any candidate model against the active v1 model before promotion.

Planned v2 notes:

- Prepare v2 manifests/configuration separately for the 50/30/20 dataset mix.
- Output v2 to `models/wav2vec2-readirect-asr-letters-v2`.
- Do not overwrite `models/wav2vec2-readirect-asr`.
- Do not switch runtime to v2 until training has completed and v2 has passed validation against v1.

## 11. API Response Fields

Important response fields include:

- `transcript`: backward-compatible displayed transcript.
- `raw_transcript`: direct Wav2Vec2 ASR output.
- `wav2vec2_transcript`: explicit Wav2Vec2 transcript field.
- `corrected_transcript`: transcript used for scoring when correction is accepted.
- `displayed_transcript`: transcript shown to the learner.
- `expected_text`: target text from Laravel/content CSV.
- `prompt_type`: letter, word, sentence, assessment, or related prompt category.
- `accepted`: whether the response was accepted by scoring/correction logic.
- `raw_wer`: WER before correction.
- `corrected_wer`: WER after accepted correction.
- `raw_cer`: CER before correction.
- `corrected_cer`: CER after accepted correction.
- `audio_quality`: audio quality metadata.
- `pause_metrics`: pause and silence metadata.
- `retry_required`: whether the learner should record again.
- `uncertain`: whether the result should be treated cautiously.
- `correction_strategy_used`: correction path used, if any.
- `debug_metadata`: additional diagnostic metadata for development/admin review.

Relevant integration documentation:

- `docs/FASTAPI_SERVICE.md`
- `docs/LARAVEL_INTEGRATION_CONTRACT.md`

## 12. Health / Status Endpoint

The health/status endpoint reports runtime readiness and should reflect:

- `wav2vec2_only` runtime status.
- Wav2Vec2 ASR model availability.
- Wav2Vec2 phoneme model availability.
- Active Wav2Vec2 model path and v2 metadata.
- Correction layer status.
- Reinforcement correction status.
- Audio quality validation status.
- Whisper removed/deprecated status.

Development service command:

```powershell
uvicorn api.main:app --reload --port 8001
```

Health check:

```powershell
curl http://127.0.0.1:8001/health
```

## 13. Deprecated / Removed Components

Whisper was previously used or referenced, but it has been removed from the active runtime. The current runtime is Wav2Vec2-only. Any old references to Whisper are deprecated and should not be followed for the current implementation.

Historical Whisper files or docs may remain for archive context. They do not describe the current ASR route, current sentence handling, current fallback path, or current runtime model.

## 14. Limitations

- The custom ReaDirect letter dataset uses adult speakers.
- Child-voice validation should be expanded in future work.
- The active v2 model still needs expanded child-voice validation.
- Correction memory is human-curated, not automatic learning.
- Sentence-level context/language-model decoding is not currently active unless implemented separately.
- Short letters and short words remain challenging for ASR and require careful correction safeguards.
- Audio quality problems can make ASR uncertain; retry metadata should be respected by Laravel/UI behavior.

## 15. What Is Included in Git

The repository can include source code, tests, configs, documentation, scripts, small templates, and safe metadata.

Large datasets, generated reports, checkpoints, private learner data, and model artifacts are generally excluded because of file size, licensing, privacy, and deployment hygiene.

## 16. Runtime Readiness Summary

Current runtime requirements:

- Active ASR model folder: `models/wav2vec2-readirect-asr-letters-v2`.
- Previous ASR model folder retained: `models/wav2vec2-readirect-asr`.
- Phoneme support model folder: `models/wav2vec2-phoneme`.
- FastAPI service code in `api/`.
- Core analysis package in `src/readirect_asr/`.
- Correction memory in `reinforcement-learning/` when enabled.
- Content/expected text supplied by Laravel or local content CSVs.

Validation command:

```powershell
python scripts/validate_ai_service_startup.py
```

Start local development service:

```powershell
python scripts/validate_ai_service_startup.py
powershell -ExecutionPolicy Bypass -File scripts/start_ai_service_dev.ps1
```

The service listens on:

```text
http://127.0.0.1:8001
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8001/health -UseBasicParsing
```

Contract test:

```powershell
python scripts/test_laravel_contract.py --base-url http://127.0.0.1:8001
```

Laravel remains separate and should not receive external datasets, training manifests, checkpoints, or model-training artifacts as part of normal integration.
