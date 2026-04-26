# Reports

Use this directory for local evaluation reports, charts, and experiment notes.

Do not commit reports containing private learner data, identifiable metadata, raw transcripts tied to learners, or large generated artifacts unless they have been reviewed and anonymized.

AI Phase 3 can generate `reports/speechocean762_readiness.md`. Generated reports remain ignored by Git by default.

AI Phase 4 can generate:

- `reports/asr_baseline_summary.md`
- `reports/asr_baseline_metrics.csv`
- `reports/asr_baseline_sample_summary.md`
- `reports/asr_baseline_sample_metrics.csv`

Generated reports remain ignored by Git by default except for this README.

AI Phase 5 can generate `reports/reading_analysis_summary.md`. Generated reading-analysis reports remain ignored by Git by default.

AI Phase 9 can generate `reports/finetuning_decision.md`. This decision report is generated locally and ignored by Git by default unless it has been reviewed and explicitly approved for sharing.

AI Phase 10 can generate:

- `reports/finetuned_whisper_eval.md`
- `reports/finetuned_whisper_metrics.json`

These are generated evaluation artifacts and remain ignored by Git.
