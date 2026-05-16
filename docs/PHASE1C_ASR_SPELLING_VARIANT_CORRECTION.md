# Phase 1C ASR Spelling-Variant Correction

## Why Phase 1C Was Needed

Phase 1 Dynamic Expected-Word Correction compared expected text to the raw Wav2Vec2 transcript using exact matching, spelling similarity, phoneme similarity, GOP evidence, homophones, and expected context. That still failed when the raw transcript was a noisy spelling approximation rather than a real intended word.

Example:

- Expected: `hand`
- Raw ASR: `hund`

The learner may have said `hand`, while Wav2Vec2 rendered the vowel as a rough spelling. Phase 1C treats this as an ASR spelling-variant problem instead of assuming `hund` is the learner's intended answer.

## Core Signals

Phase 1C adds a `dynamic_asr_spelling_variant` layer inside the existing correction pipeline.

It checks:

- normalized edit similarity
- consonant skeleton similarity
- vowel-tolerant similarity
- expected phoneme coverage
- GOP score when available
- expected-word context

Consonant skeleton matching removes vowels and compares consonant structure:

- `hand` -> `hnd`, `hund` -> `hnd`
- `shield` -> `shld`, `shild` -> `shld`
- `tree` -> `tr`, `tr` -> `tr`

Vowel-tolerant matching gives vowel substitutions a lower penalty than consonant substitutions. This makes `hand / hund` plausible while keeping `hand / head` risky unless stronger GOP or phoneme evidence supports `hand`.

## Accepted Examples

- `hand / hund`: accepted as `hand` by vowel-tolerant consonant skeleton matching.
- `shield / shild`: accepted as `shield`.
- `Leo / Layo`: still accepted when GOP or phoneme evidence supports it.

## Fragment Safety

Phase 1C keeps the pre-Phase 1B fragment safeguards.

These do not pass through spelling or consonant skeleton alone:

- `fish / fs`
- `tree / tr`

They are accepted only when strong GOP or phoneme evidence supports the missing vowel or ending sounds. Bad, clipped, retry-required, or mostly silent audio still requires retry.

## Rejected Examples

- `hand / banana`: rejected as unrelated.
- `hand / head`: rejected without strong GOP or phoneme evidence.
- `the / a`: rejected by short function-word safety.

## Transcript Contract

The transcript contract is unchanged:

- `raw_transcript` remains the direct Wav2Vec2 output.
- `corrected_transcript` becomes the expected word only when evidence is strong.
- `displayed_transcript` becomes the expected word for accepted isolated word/rhyme/letter prompts.
- Full sentence, paragraph, and passage displayed transcripts are not forced to expected text.

For sentence and passage prompts, Phase 1C only applies to clean one-to-one aligned word pairs that already exist in the current alignment. It can mark a word as `accepted_by_asr_spelling_variant`, but it does not repair split/merge alignment problems.

## API Metadata

Responses may include:

- `asr_spelling_variant_enabled`
- `asr_spelling_variant_applied`
- `asr_spelling_variant_strategy`
- `asr_spelling_variant_sub_strategy`
- `asr_spelling_variant_confidence`
- `asr_spelling_variant_threshold`
- `consonant_skeleton_similarity`
- `vowel_tolerant_similarity`
- `expected_phoneme_coverage`
- `variant_edit_similarity`
- `variant_reason`

## Future Work

Phase 1B will handle passage-aware split/merge alignment repair, such as:

- `time after` vs `timeafter`
- `open Maya` vs `openmya`
- `fruit seeds` vs `fruitsieds`

Beam search and vocabulary-constrained rescoring remain later phases and were not implemented here.
