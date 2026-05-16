# GOP Pronunciation Evidence

GOP in ReaDirect is a practical Goodness of Pronunciation style evidence layer. It is not a second transcript and it does not replace the raw Wav2Vec2 ASR output.

## Purpose

Wav2Vec2 can sometimes transcribe a correctly pronounced short target as a similar wrong-looking word. GOP reduces these false ASR errors by comparing expected phonemes with observed phoneme evidence from `models/wav2vec2-phoneme`.

Example:

```text
expected_text: Leo
raw_transcript: Layo
observed phonemes: L EY OW
expected phonemes: L IY OW
GOP score: 0.82
decision: accepted_by_pronunciation_evidence
```

For a short word prompt, this allows:

```text
corrected_transcript = Leo
displayed_transcript = Leo
accepted = true
correction_strategy_used = gop_pronunciation_evidence
```

`raw_transcript` remains `Layo`.

## Transcript Separation

GOP is returned separately from transcript fields.

- `raw_transcript`: direct Wav2Vec2 output.
- `corrected_transcript`: scoring transcript when a correction is accepted.
- `displayed_transcript`: learner-facing transcript.
- GOP fields: pronunciation metadata only.

GOP data must not be inserted into `raw_transcript`, `corrected_transcript`, or `displayed_transcript`.

## Expected Phonemes

Expected phonemes are generated in this order:

1. ReaDirect letter mappings and word overrides.
2. CMUdict.
3. `g2p_en`, when installed.
4. Simple fallback approximation for limited cases where no dictionary entry exists.

Letter prompts use American English names. `Z` is treated as `Z IY`.

## Observed Phonemes

Observed phonemes come from `models/wav2vec2-phoneme` using CTC decoding. If a provider already returned `observed_phonemes`, GOP uses those values. If a phoneme model and processor are supplied directly, GOP can decode phonemes from audio.

## Scoring

The composite GOP score is normalized to `0.0..1.0`:

```text
0.50 * phoneme_sequence_similarity
0.30 * phoneme_alignment_score
0.15 * acoustic_confidence_score
0.05 * transcript_support_score
```

When acoustic confidence is not available, the implementation uses observed phoneme alignment quality as practical confidence evidence.

## Thresholds

Defaults:

```text
GOP_ENABLED=true
GOP_LETTER_THRESHOLD=0.70
GOP_WORD_THRESHOLD=0.75
GOP_RHYME_THRESHOLD=0.75
GOP_SENTENCE_WORD_THRESHOLD=0.70
GOP_PASSAGE_WORD_THRESHOLD=0.70
GOP_MIN_AUDIO_QUALITY_REQUIRED=true
GOP_SKIP_ON_RETRY_REQUIRED=true
GOP_SKIP_ON_UNCERTAIN_AUDIO=true
GOP_DEBUG=false
```

Decision labels:

- `accepted_by_pronunciation_evidence`
- `rejected_low_gop`
- `not_available`
- `skipped_bad_audio`
- `skipped_no_expected_text`
- `skipped_unsupported_prompt_type`
- `error`

## Short Prompt Behavior

For letters, words, and rhyming prompts, GOP can accept the expected target when the normal correction pipeline rejected the raw transcript but pronunciation evidence is strong enough.

When accepted, the response sets:

- `accepted = true`
- `corrected_transcript = expected_text`
- `displayed_transcript = expected_text`
- `correction_strategy_used = gop_pronunciation_evidence`
- `gop_correction_applied = true`

## Sentence And Passage Behavior

GOP does not force full sentence, paragraph, or passage transcripts to `expected_text`.

For longer prompts, GOP is pronunciation evidence only:

- `gop_word_scores`
- `gop_phoneme_scores`
- `weak_words`
- `mispronounced_phonemes`

Sentence and passage scoring may use this metadata as word-level evidence, but the displayed transcript remains the ASR transcript/corrected transcript from the normal sentence pipeline.

## API Fields

The API returns:

- `gop_enabled`
- `gop_available`
- `gop_score`
- `gop_confidence`
- `gop_decision`
- `gop_threshold`
- `gop_prompt_type`
- `gop_expected_phonemes`
- `gop_observed_phonemes`
- `gop_phoneme_scores`
- `gop_word_scores`
- `mispronounced_phonemes`
- `weak_words`
- `gop_correction_applied`
- `gop_error`

When `GOP_DEBUG=true`, GOP details are also included under `debug_metadata.gop`.

## Safety Rules

GOP cannot accept audio when:

- `retry_required = true`
- audio quality flags indicate silence, no speech, too short, or clipping
- `uncertain = true` and `GOP_SKIP_ON_UNCERTAIN_AUDIO=true`
- `expected_text` is missing
- observed phoneme evidence is unavailable

In those cases the ASR request still returns safely, with a skipped or not-available GOP decision.

## Laravel Interpretation

Laravel does not calculate GOP. It stores and displays the GOP fields returned by FastAPI.

Learner-facing copy should stay simple, for example:

```text
Your pronunciation was close to the target word.
```

Admin/developer debug can show GOP score, decision, threshold, expected phonemes, observed phonemes, weak words, and correction strategy.

## Limitations

This is GOP-style pronunciation evidence, not a full forced-alignment GOP engine. It uses CTC phoneme decoding, sequence similarity, alignment scoring, and confidence proxies available in the current Wav2Vec2-only runtime.
