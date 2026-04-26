# Data Directory

Use this directory for local datasets during ASR experiments.

## Allowed to Track

- This README.
- `.gitkeep` placeholders.
- Small synthetic examples only if they contain no learner data.
- `data/manifests/metadata_template.csv`, which contains fake example rows only.

## Do Not Commit

- Real learner audio.
- Identifiable learner metadata.
- Raw private datasets.
- Generated audio.
- Large manifests or annotation exports containing private fields.

Use anonymized learner IDs and keep raw data local or in approved secure storage.

Generated `dataset_manifest.csv` and `content_index.csv` files are local working artifacts and are ignored by Git by default.

AI Phase 3 also generates:

- `data/manifests/speechocean762_manifest.csv`
- `data/manifests/unified_public_dataset_manifest.csv`

These are ignored by Git because they are generated from external datasets.

AI Phase 4 also generates:

- `data/manifests/speechocean762_asr_baseline.csv`
- `data/manifests/*_asr_baseline.csv`

These contain ASR outputs and are ignored by Git.

AI Phase 5 also generates:

- `data/manifests/speechocean762_reading_analysis.csv`
- `data/manifests/*_reading_analysis.csv`

These contain derived reading-analysis outputs and are ignored by Git.

AI Phase 9 can generate:

- `data/processed/whisper_finetune/train.jsonl`
- `data/processed/whisper_finetune/validation.jsonl`
- `data/processed/whisper_finetune/test.jsonl`
- `data/processed/whisper_finetune/dataset_summary.json`

These are local training-preparation artifacts and are ignored by Git.
