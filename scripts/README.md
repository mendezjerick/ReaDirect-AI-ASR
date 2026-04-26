# Scripts

Command-line utilities for dataset inspection, manifest creation, audio preparation, validation, and exports for Laravel integration live here.

Scripts should be safe by default:

- Do not destructively modify raw audio.
- Do not upload private data.
- Do not write model checkpoints.
- Prefer explicit input and output paths.

Phase 2 dataset commands:

```powershell
python scripts/inspect_content_bank.py --content-bank content_bank --cmudict external_datasets/cmudict
python scripts/build_content_index.py --content-bank content_bank --cmudict-dir external_datasets/cmudict --output data/manifests/content_index.csv
python scripts/build_manifest.py --metadata-csv data/manifests/metadata_template.csv --content-index data/manifests/content_index.csv --audio-dir data/raw --output data/manifests/dataset_manifest.csv
python scripts/validate_manifest.py --manifest data/manifests/dataset_manifest.csv --content-index data/manifests/content_index.csv --audio-base data/raw
```

Phase 3 Speechocean762 commands:

```powershell
python scripts/inspect_speechocean762.py --dataset-dir external_datasets/speechocean762 --print-tree
python scripts/extract_speechocean762.py --archive external_datasets/speechocean762/raw/speechocean762.tar.gz --dest external_datasets/speechocean762/extracted
python scripts/build_speechocean762_manifest.py --dataset-dir external_datasets/speechocean762/extracted --cmudict-dir external_datasets/cmudict --output data/manifests/speechocean762_manifest.csv
python scripts/build_public_dataset_manifest.py --speechocean-manifest data/manifests/speechocean762_manifest.csv --output data/manifests/unified_public_dataset_manifest.csv
python scripts/report_dataset_readiness.py --manifest data/manifests/speechocean762_manifest.csv --output reports/speechocean762_readiness.md
```

Phase 4 ASR baseline commands:

```powershell
python scripts/run_phase4_sample.py --limit 5 --model-size base.en --device cpu --compute-type int8
python scripts/run_asr_baseline.py --manifest data/manifests/speechocean762_manifest.csv --output data/manifests/speechocean762_asr_baseline.csv --model-size base.en --device cpu --compute-type int8 --limit 50
python scripts/evaluate_asr_baseline.py --input data/manifests/speechocean762_asr_baseline.csv --output reports/asr_baseline_summary.md --metrics-csv reports/asr_baseline_metrics.csv
```

Phase 5 reading-analysis commands:

```powershell
python scripts/analyze_asr_outputs.py --input data/manifests/speechocean762_asr_baseline.csv --output data/manifests/speechocean762_reading_analysis.csv --cmudict-dir external_datasets/cmudict
python scripts/report_reading_analysis.py --input data/manifests/speechocean762_reading_analysis.csv --output reports/reading_analysis_summary.md
```

Phase 6 content-enrichment commands:

```powershell
python scripts/enrich_content_bank.py --content-bank content_bank --content-index data/manifests/content_index.csv --cmudict-dir external_datasets/cmudict --output-dir content_bank_enriched --write-import-ready
python scripts/validate_enriched_content.py --input content_bank_enriched/enriched_content_index.csv
python scripts/report_content_enrichment.py --enriched-index content_bank_enriched/enriched_content_index.csv --output content_bank_enriched/reports/content_enrichment_report.md
python scripts/export_enriched_content_zip.py --source-dir content_bank_enriched/import_ready --output content_bank_enriched/readirect-enriched-content.zip
```

Phase 7 API test commands:

```powershell
uvicorn api.main:app --reload --port 8001
python scripts/test_api_analysis.py --mode text --expected-text cat --actual-text cap --accepted-answer cat --debug
python scripts/test_api_analysis.py --mode audio --audio-path data/samples/sample.wav --expected-text cat --accepted-answer cat --debug
```

Phase 8 adaptive tutoring commands:

```powershell
python scripts/simulate_adaptive_tutoring.py --top-k 5
```

The simulation uses built-in fake learner histories when `--history-csv` is not provided. Generated reports under `reports/` are ignored by Git.

Phase 9 fine-tuning decision commands:

```powershell
python scripts/decide_finetuning.py --manifest data/manifests/speechocean762_manifest.csv --baseline data/manifests/speechocean762_asr_baseline.csv --output reports/finetuning_decision.md
python scripts/prepare_whisper_finetune_dataset.py --manifest data/manifests/speechocean762_manifest.csv --output-dir data/processed/whisper_finetune
python training/train_whisper_skeleton.py --config configs/whisper_finetune_config.yaml
```

These commands do not train a model. They generate a decision report, prepare JSONL files, and preview a future training config.

Phase 10 guarded Whisper fine-tuning commands:

```powershell
python scripts/check_training_environment.py
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --dry-run
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --run
python scripts/evaluate_finetuned_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --test-jsonl data/processed/whisper_finetune/test.jsonl --output reports/finetuned_whisper_eval.md --metrics-json reports/finetuned_whisper_metrics.json
python scripts/convert_whisper_to_faster_whisper.py --model-dir model_artifacts/readirect-whisper-base-en-v1-hf --output-dir model_artifacts/readirect-whisper-base-en-v1-ct2 --quantization int8_float16 --dry-run
```

Training only starts on the `--run` command.
