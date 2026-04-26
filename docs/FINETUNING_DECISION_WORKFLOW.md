# Fine-Tuning Decision Workflow

Phase 9 decides whether Whisper fine-tuning is justified. It does not train a model.

Baseline evaluation comes first because fine-tuning can waste time, introduce maintenance burden, and overfit if the pretrained model already handles ReaDirect utterances well enough.

## Required Inputs

- `data/manifests/speechocean762_manifest.csv`
- `data/manifests/speechocean762_asr_baseline.csv`
- Manual transcripts or expected text references
- Audio paths and duration metadata where available

## Metrics Used

- WER
- CER
- Exact match rate
- Blank hypothesis rate
- ReaDirect short-word exact match
- ReaDirect short-word CER
- Common short-word confusions

Short-word accuracy matters because early ReaDirect tasks rely on words such as `cat`, `dog`, `sun`, `pen`, `map`, `cup`, `hat`, `pig`, `run`, and `box`.

## Dataset Readiness Rules

Default thresholds:

- Minimum rows: 500
- Minimum labeled audio: 2 hours
- Transcript coverage: 90%

The readiness checker also reports missing audio, duplicate audio paths, blank transcripts, duration issues, and baseline availability.

## Decision Thresholds

Default rules:

- WER > 0.20 and CER > 0.10: fine-tuning recommended.
- Short-word exact match < 0.75: fine-tuning recommended.
- Blank hypothesis rate > 0.05: fine-tuning recommended.
- WER <= 0.10, CER <= 0.05, and short-word accuracy >= 0.85: not needed yet.
- Missing baseline: run baseline first.
- Not enough data: collect or convert more labeled data first.

Thresholds live in `configs/finetuning_decision.yaml`.

## Commands

```powershell
python scripts/decide_finetuning.py --manifest data/manifests/speechocean762_manifest.csv --baseline data/manifests/speechocean762_asr_baseline.csv --output reports/finetuning_decision.md
python scripts/prepare_whisper_finetune_dataset.py --manifest data/manifests/speechocean762_manifest.csv --output-dir data/processed/whisper_finetune
python training/train_whisper_skeleton.py --config configs/whisper_finetune_config.yaml
```

## Outputs

- `reports/finetuning_decision.md`
- `data/processed/whisper_finetune/train.jsonl`
- `data/processed/whisper_finetune/validation.jsonl`
- `data/processed/whisper_finetune/test.jsonl`
- `data/processed/whisper_finetune/dataset_summary.json`

These outputs are generated artifacts and are ignored by Git.

## Interpretation

Possible decisions:

- `fine_tuning_recommended`
- `not_needed_yet`
- `more_data_needed`
- `baseline_missing`

Fine-tuning should only proceed later if the report shows a clear need and the dataset is license-safe, privacy-safe, and sufficiently large.
