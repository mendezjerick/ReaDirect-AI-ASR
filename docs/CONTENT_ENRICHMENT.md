# Content Enrichment

AI Phase 6 turns curated ReaDirect CSV rows into AI-ready learning content.

## Output Schema

Enriched rows include:

- identity fields: `prompt_id`, `source_file`, `source_group`, `module_key`, `activity_type`, `expected_text`
- phoneme metadata: `expected_phonemes`, `initial_phoneme`, `vowel_phonemes`, `final_phoneme`, `phoneme_pattern`, `phoneme_count`
- skill metadata: `skill_tag`, `skill_group`, `error_focus`, `target_position`, `target_phoneme`, `target_grapheme`
- adaptive metadata: `adaptive_bucket`, `recommended_for_error_type`, `remediation_priority`, `practice_role`, `mastery_candidate`, `review_candidate`
- quality metadata: `enrichment_status`, `enrichment_warnings`, `needs_manual_review`

## Phoneme Metadata

CMUdict provides dictionary phonemes for words and sentences. Letter items use a simple letter-sound mapping. Missing words are recorded in `cmudict_missing_words` and marked for manual review.

## Skill Tag Rules

- Module 1: `letter_sound`
- Module 2: `word_reading`, CVC tags, word-family tags, initial/final/vowel focus
- Module 3: `sentence_reading`, `sentence_tracking`, `fluency_pacing`
- Reading passages: `fluency`
- Comprehension questions: `comprehension`

## Difficulty Scoring

Difficulty is heuristic. It considers:

- phoneme count
- syllable estimate
- word length
- blends/digraphs
- sentence word count
- average word length
- punctuation
- CMUdict missing-word risk

Levels:

- `very_easy`
- `easy`
- `medium`
- `hard`
- `very_hard`

## Adaptive Metadata

Adaptive metadata maps item focus to learner needs:

- `final_consonant -> final_sound_error`
- `initial_consonant -> initial_sound_error`
- `vowel_sound -> vowel_error`
- `sentence_tracking -> skipped_word`
- `fluency_pacing -> partial_sentence`

## Manual Review Flags

Rows should be reviewed when:

- CMUdict lookup is missing,
- phoneme enrichment is minimal,
- source content has no expected answer,
- difficulty or skill tags look inconsistent with educator intent.

## Commands

```powershell
python scripts/enrich_content_bank.py --content-bank content_bank --content-index data/manifests/content_index.csv --cmudict-dir external_datasets/cmudict --output-dir content_bank_enriched --write-import-ready
python scripts/validate_enriched_content.py --input content_bank_enriched/enriched_content_index.csv
python scripts/report_content_enrichment.py --enriched-index content_bank_enriched/enriched_content_index.csv --output content_bank_enriched/reports/content_enrichment_report.md
python scripts/export_enriched_content_zip.py --source-dir content_bank_enriched/import_ready --output content_bank_enriched/readirect-enriched-content.zip
```

## Import-Back Workflow

The ZIP from `content_bank_enriched/import_ready` is for review. It does not modify the Laravel repository. Review enriched columns, educator-facing assumptions, and manual-review rows before importing into the main ReaDirect app.

## Limitations

- CMUdict is an American English pronunciation dictionary.
- Names, local terms, or early-grade invented words may be missing.
- Phoneme tags are dictionary-derived, not acoustic.
- Difficulty scoring is heuristic and should be reviewed by educators.

