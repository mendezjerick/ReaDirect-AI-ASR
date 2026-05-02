# Reinforcement Corrections

The `reinforcement-learning/` folder is a human-curated correction memory for known ASR transcript errors. It is not model training, not automatic self-learning, and not true reinforcement learning in the runtime behavior.

The correction layer reads CSV rows that pair an expected label with an observed ASR transcript error. When the active prompt expects that label and the raw Wav2Vec2 transcript matches the observed error, the service accepts the response and displays the expected label.

## Letter CSV Format

`reinforcement-learning/letter-reinforcement.csv`

```csv
letter,transcript-error
Z,They
O,Oh
C,See
```

For future word-level files, the loader also supports these column pairs:

```csv
expected,transcript-error
word,transcript-error
expected_text,raw_transcript
```

Rows are matched case-insensitively with whitespace normalization. Invalid rows are skipped with load warnings.

## Safety Rule

Corrections are expected-centric. A transcript error only applies when `expected_text` matches the CSV label.

If `expected_text` is `Z` and `raw_transcript` is `They`, the service accepts and displays `Z`.

If `expected_text` is `C` and `raw_transcript` is `They`, the service does not correct to `Z`.

Letter and word reinforcement rules are not applied to full sentence prompts, and rejected cases keep the recognized transcript for display.

## Configuration

```env
REINFORCEMENT_CORRECTIONS_ENABLED=true
REINFORCEMENT_CORRECTIONS_DIR=reinforcement-learning
LETTER_REINFORCEMENT_FILE=letter-reinforcement.csv
```

If the feature is disabled, the correction layer skips the curated table and continues with the existing alias, confusion, phoneme, threshold, and WER/CER behavior. If the folder or CSV is missing, startup/status metadata reports a warning and the service continues.

## API Metadata

Accepted reinforcement matches set:

```json
{
  "accepted_by_reinforcement_match": true,
  "correction_strategy_used": "reinforcement_error_transcript_match",
  "reinforcement_source_file": "reinforcement-learning/letter-reinforcement.csv",
  "reinforcement_expected_label": "Z",
  "reinforcement_matched_transcript": "They"
}
```

The `/health` endpoint reports whether the table is enabled, which files loaded, rule counts, and load warnings.

## Manual Verification

Run:

```powershell
python scripts/verify_correction_contract.py
```

The script includes reinforcement cases such as `Z / They`, `O / Oh`, `C / See`, `Q / Cue`, `W / Zavil you`, `W / Zebel you`, plus rejected checks for `Z / Banana` and `C / They`.
