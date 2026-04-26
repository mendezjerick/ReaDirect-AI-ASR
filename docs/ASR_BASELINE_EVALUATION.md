# ASR Baseline Evaluation

AI Phase 4 evaluates pretrained faster-whisper on Speechocean762 before any fine-tuning.

## Inputs

- `data/manifests/speechocean762_manifest.csv`
- Audio paths from the extracted Speechocean762 dataset.
- Reference transcript from `manual_transcript` or `expected_text`.

## Outputs

- `data/manifests/speechocean762_asr_baseline.csv`
- `reports/asr_baseline_summary.md`
- `reports/asr_baseline_metrics.csv`

Generated manifests and reports are ignored by Git.

## Workflow

```powershell
pip install -r requirements.txt
python scripts/run_phase4_sample.py --limit 5 --model-size base.en --device cpu --compute-type int8
python scripts/run_asr_baseline.py --manifest data/manifests/speechocean762_manifest.csv --output data/manifests/speechocean762_asr_baseline.csv --model-size base.en --device cpu --compute-type int8 --limit 50
python scripts/evaluate_asr_baseline.py --input data/manifests/speechocean762_asr_baseline.csv --output reports/asr_baseline_summary.md --metrics-csv reports/asr_baseline_metrics.csv
```

## Metrics

- WER: word error rate. Lower is better.
- CER: character error rate. Lower is better and especially useful for short words.
- Exact match rate: normalized transcript equals normalized reference.
- Token accuracy: proportion of reference tokens matched by the hypothesis.

## Common Problems

- `faster-whisper is not installed`: run `pip install -r requirements.txt` or `pip install faster-whisper`.
- First model run is slow: faster-whisper may download model files into a local cache.
- CPU runs are slow: use `base.en`, `int8`, and a small `--limit` first.
- Blank ASR outputs: check audio paths, file readability, and model errors in `asr_error`.

## Fine-Tuning Signals

Fine-tuning may be needed if:

- short-word CER is high,
- exact match is poor for simple utterances,
- common substitutions affect ReaDirect expected-answer checks,
- child/adult or age-group breakdowns show strong performance gaps.

Use the Phase 9 decision script before starting any training:

```powershell
python scripts/decide_finetuning.py --manifest data/manifests/speechocean762_manifest.csv --baseline data/manifests/speechocean762_asr_baseline.csv --output reports/finetuning_decision.md
```
