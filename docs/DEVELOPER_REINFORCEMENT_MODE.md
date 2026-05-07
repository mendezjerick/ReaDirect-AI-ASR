# Developer Reinforcement Mode

Developer reinforcement mode is a correction-memory workflow for ReaDirect ASR. It is not model training and does not update Wav2Vec2 weights. When enabled by Laravel for an authenticated admin/developer, incorrect ASR outputs can be written to CSV correction memory and immediately reloaded by the existing transcript correction layer.

## Admin-Only Behavior

Laravel owns the admin toggle and only sends `developer_reinforcement_enabled: true` for authenticated admin/developer users. Learner requests must never send this flag. The AI service still validates that developer mode is enabled and that the caller role is `admin`, `developer`, `system_admin`, or `school_admin` before writing.

## CSV Locations

- `reinforcement-learning/letter-reinforcement.csv`
- `reinforcement-learning/word-reinforcement.csv`
- `reinforcement-learning/reinforcement-audit.log`

Both CSV files use:

```csv
expected_text,raw_transcript,normalized_expected,normalized_raw,prompt_type,source,created_at,created_by,notes
```

Older CSV columns such as `letter,transcript-error` and `word,transcript-error` remain readable and are migrated when the file is written.

## Routing Rules

Only `letter` prompts write to and read from `letter-reinforcement.csv`.

These prompt types write to and read from `word-reinforcement.csv`:

- `word`
- `rhyme`
- `rhyming_word`
- `sentence`
- `paragraph`
- `passage`
- `final_sentence`
- `reading_passage`

Correction lookup is expected-centric. A row for `C` plus `See` only corrects `See` when the current expected text is `C`; it will not correct `See` for expected `Z`.

## Enhancement Cycle

1. Developer turns Developer Reinforcement Mode ON.
2. Developer opens a letter, word, rhyme, sentence, paragraph, or passage activity.
3. Developer speaks the expected answer.
4. ASR produces `raw_transcript`.
5. If `raw_transcript` is wrong and `expected_text` is known, the system writes the pair into the correct CSV.
6. AI correction memory reloads when the CSV modified time changes or immediately after append.
7. Developer repeats the same item.
8. The correction layer reads the new CSV row.
9. `corrected_transcript` and `displayed_transcript` become the expected answer.
10. Developer turns Developer Reinforcement Mode OFF before normal learner testing.

## Examples

Letter:

```csv
C,See,c,see,letter,developer_auto,2026-05-07T00:00:00Z,admin,"auto-added from developer reinforcement mode"
```

Word:

```csv
Leo,Layo,leo,layo,word,developer_auto,2026-05-07T00:00:00Z,admin,"auto-added from developer reinforcement mode"
```

## Safety Rules

The AI service skips writes when expected text is empty, raw transcript is empty, normalized raw equals normalized expected, audio is retry-required, audio is uncertain, the prompt type is unsupported, the caller is not admin/developer, developer mode is off, or the same correction pair already exists.

It only appends when the transcript was not accepted or the correction strategy indicates no correction was found.

## API

Manual append endpoint:

```http
POST /reinforcement/corrections
```

```json
{
  "expected_text": "C",
  "raw_transcript": "See",
  "prompt_type": "letter",
  "accepted": false,
  "retry_required": false,
  "uncertain": false,
  "created_by": "admin",
  "source": "developer_auto"
}
```

Auto-write during `/analyze-audio` uses:

```json
{
  "developer_reinforcement_enabled": true,
  "developer_user_role": "admin",
  "developer_user_id": "admin@example.com"
}
```

## Verification

Turn mode ON in Laravel, record a known wrong case, then inspect the target CSV and `reinforcement-audit.log`. Repeat the same prompt and confirm the response returns the expected answer in `corrected_transcript` and `displayed_transcript`.

No service restart is normally required because the correction table reloads when file modified times change and the append path clears the cache.

Turn mode OFF before learner testing.
