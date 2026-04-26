# Reading Analysis Engine

AI Phase 5 turns expected answers, accepted answers, ASR transcripts, and CMUdict phoneme tags into ReaDirect reading-analysis signals.

## Pipeline

```text
expected answer
accepted answers
ASR transcript
CMUdict phonemes
-> answer matching
-> text similarity
-> transcript-derived phoneme comparison
-> heuristic error detection
-> feedback hint
-> adaptive skill signal
```

## Expected Answer Matching

The answer matcher normalizes case, punctuation, and whitespace. Exact matches and accepted-answer matches are marked correct. Close answers are not automatically marked correct because official scoring remains rule-based in Laravel.

## Accepted Answers

Accepted answers can be provided as a list, JSON list, comma-separated string, or pipe-separated string.

## Text Similarity

Similarity outputs include:

- `character_similarity`
- `token_similarity`
- `similarity_label`

Labels:

- `exact`
- `very_close`
- `close`
- `somewhat_close`
- `far`
- `blank`

Short early-reading words get special handling so one-letter differences like `cat` vs `cap` can still be labeled `very_close` for feedback.

## Phoneme Comparison

CMUdict maps expected text and ASR transcript text to ARPABET phonemes. The engine compares:

- initial phoneme
- final phoneme
- vowel phonemes
- phoneme edit distance
- phoneme similarity

This is transcript-derived phoneme comparison, not acoustic phoneme recognition.

## Error Type Rules

Supported error types include:

- `correct`
- `accepted_variant`
- `blank`
- `very_close_text_error`
- `initial_sound_error`
- `final_sound_error`
- `vowel_error`
- `consonant_error`
- `substitution`
- `omission`
- `insertion`
- `skipped_word`
- `word_order_error`
- `partial_sentence`
- `far_answer`
- `unclear_asr`
- `incorrect_general`

## Feedback Hints

Error types map to safe template keys, such as:

- `final_sound_error -> listen_to_final_sound`
- `initial_sound_error -> listen_to_first_sound`
- `vowel_error -> listen_to_middle_sound`
- `skipped_word -> read_each_word`

The AI repo returns stable keys and short safe summaries. Laravel or a Coach Agent can decide how to present them.

## Adaptive Skill Signals

Skill signals include:

- `initial_consonant`
- `final_consonant`
- `vowel_sound`
- `sentence_tracking`
- `fluency_completion`
- `sentence_order`

These are intended to help later item selection from the ReaDirect content bank.

AI Phase 6 enriches content-bank rows with matching skill tags and target phonemes so these signals can be connected to actual practice items.

## Example

Expected: `cat`

Actual: `cap`

Result:

```text
similarity_label = very_close
error_type = final_sound_error
skill_signal = final_consonant
feedback_hint = ending_sound
```

## Limitations

- Actual phonemes are derived from the ASR transcript, not direct acoustic phoneme recognition.
- ASR mistakes can affect error detection.
- This is not yet a trained pronunciation model.
- Speechocean annotations may later improve scoring calibration.
- Official scoring still belongs to Laravel.
