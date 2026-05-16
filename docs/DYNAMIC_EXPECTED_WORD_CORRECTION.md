# Dynamic Expected-Word Correction

## Purpose

Dynamic expected-word correction is a non-LLM correction layer for ReaDirect ASR. It asks whether the raw greedy Wav2Vec2 transcript could reasonably be the expected word, even when the spelling is wrong.

The goal is to reduce false ASR errors such as:

- expected `shield`, raw `shild`
- expected `knights`, raw `nights`
- expected `Leo`, raw `Layo`
- expected `C`, raw `See`

It does not train a model, call an LLM, use OpenAI, or change ASR weights.

## Transcript Contract

The existing transcript contract remains unchanged:

- `raw_transcript`: direct greedy Wav2Vec2 output, never overwritten
- `corrected_transcript`: transcript accepted for scoring when correction is safe
- `displayed_transcript`: transcript shown to the learner
- `expected_text`: target answer from Laravel/content bank

For isolated letters, words, rhymes, and short word prompts, dynamic correction may set `corrected_transcript` and `displayed_transcript` to `expected_text`.

For sentences, paragraphs, and passages, dynamic correction does not force the full transcript to the expected text. It annotates word-level alignment metadata only.

## How It Differs From Correction Memory

Developer reinforcement correction memory uses curated rows from:

- `reinforcement-learning/letter-reinforcement.csv`
- `reinforcement-learning/word-reinforcement.csv`

Dynamic expected-word correction does not require a curated pair to already exist. It computes evidence on the fly from the current expected word and raw ASR output.

Reinforcement memory remains useful, but dynamic correction handles new words such as `shield` / `shild` without needing a CSV entry.

## Algorithm Overview

For a candidate expected/raw pair, the layer computes:

1. Exact normalized match
2. Spelling similarity
3. Phoneme similarity from CMUdict/g2p/fallback phoneme utilities
4. GOP score when available
5. Homophone or near-homophone match
6. Expected-position/context score
7. Known ASR confusion support
8. Safety gates for retry-required, uncertain audio, missing expected text, and short function words

The correction is accepted only when the combined evidence is strong enough for the prompt type.

## Scoring Formula

For isolated letters:

```text
0.35 * phoneme_similarity
+ 0.25 * gop_score
+ 0.20 * spelling_similarity
+ 0.10 * known_confusion_score
+ 0.10 * context_score
```

For isolated words and rhymes:

```text
0.30 * phoneme_similarity
+ 0.25 * gop_score
+ 0.20 * spelling_similarity
+ 0.15 * context_score
+ 0.10 * known_confusion_score
```

For sentence, paragraph, and passage word-level alignment:

```text
0.30 * phoneme_similarity
+ 0.20 * gop_score
+ 0.20 * spelling_similarity
+ 0.20 * expected_position_context_score
+ 0.10 * known_confusion_score
```

Unavailable GOP or phoneme signals are omitted and the remaining available weights are normalized safely. Spelling-only acceptance requires very high spelling similarity and strong expected context.

## Thresholds

Config/env defaults:

```text
DYNAMIC_EXPECTED_CORRECTION_ENABLED=true
DYNAMIC_LETTER_ACCEPT_THRESHOLD=0.72
DYNAMIC_WORD_ACCEPT_THRESHOLD=0.78
DYNAMIC_RHYME_ACCEPT_THRESHOLD=0.78
DYNAMIC_SENTENCE_WORD_ACCEPT_THRESHOLD=0.80
DYNAMIC_PASSAGE_WORD_ACCEPT_THRESHOLD=0.82
DYNAMIC_HOMOPHONE_THRESHOLD=0.96
DYNAMIC_MIN_PHONEME_FOR_LOW_TEXT_MATCH=0.90
DYNAMIC_MIN_GOP_FOR_ACCEPTANCE=0.75
DYNAMIC_SKIP_ON_RETRY_REQUIRED=true
DYNAMIC_SKIP_ON_UNCERTAIN_AUDIO=true
DYNAMIC_DEBUG=false
```

## Examples

### `shield` / `shild`

`shield` and `shild` have high spelling similarity. With strong expected context, the layer accepts `shield`.

Result:

```json
{
  "accepted": true,
  "corrected_transcript": "shield",
  "displayed_transcript": "shield",
  "dynamic_correction_sub_strategy": "spelling_context_expected_match"
}
```

### `knights` / `nights`

The phoneme sequences are identical or near-identical, so the pair is accepted as a homophone.

Result:

```json
{
  "accepted": true,
  "dynamic_homophone_match": true,
  "dynamic_correction_sub_strategy": "homophone_match"
}
```

### `Leo` / `Layo`

When GOP pronunciation evidence is strong and text/context evidence is plausible, the layer accepts `Leo`.

Result:

```json
{
  "accepted": true,
  "corrected_transcript": "Leo",
  "displayed_transcript": "Leo",
  "dynamic_gop_score": 0.84,
  "dynamic_correction_sub_strategy": "gop_supported_expected_match"
}
```

### `C` / `See`

Letter aliases allow spoken letter names to map back to the expected letter.

Result:

```json
{
  "accepted": true,
  "corrected_transcript": "C",
  "displayed_transcript": "C"
}
```

### `Leo` / `banana`

The spelling, phoneme, and GOP evidence are weak or missing.

Result:

```json
{
  "accepted": false,
  "displayed_transcript": "banana",
  "dynamic_correction_sub_strategy": "rejected_low_similarity"
}
```

## Sentence and Passage Rule

For sentence, paragraph, passage, `reading_passage`, `final_sentence`, and `final_passage` prompts:

- the full `corrected_transcript` is not forced to `expected_text`
- the full `displayed_transcript` is not forced to `expected_text`
- dynamic correction is applied per aligned expected/recognized word
- accepted word-level variants are marked with `counts_as_correct: true`

Example word alignment item:

```json
{
  "expected_word": "shield",
  "recognized_word": "shild",
  "status": "accepted_by_dynamic_expected_word_correction",
  "counts_as_correct": true,
  "dynamic_correction_confidence": 0.91,
  "spelling_similarity": 0.89,
  "phoneme_similarity": 0.95,
  "gop_score": 0.82,
  "sub_strategy": "spelling_phoneme_expected_match"
}
```

Homophones use:

```json
{
  "status": "accepted_by_homophone",
  "counts_as_correct": true
}
```

## API Fields

The response includes:

- `dynamic_correction_enabled`
- `dynamic_correction_applied`
- `dynamic_correction_strategy`
- `dynamic_correction_sub_strategy`
- `dynamic_correction_confidence`
- `dynamic_correction_threshold`
- `dynamic_spelling_similarity`
- `dynamic_phoneme_similarity`
- `dynamic_gop_score`
- `dynamic_homophone_match`
- `dynamic_context_score`
- `dynamic_correction_reason`
- `word_alignment`

Detailed debug payloads are also included in `debug_metadata.dynamic_expected_word_correction`.

## Safety Rules

Dynamic correction is skipped when:

- `retry_required` is true and skip-on-retry is enabled
- audio is uncertain and skip-on-uncertain is enabled
- `expected_text` is missing
- raw transcript is blank
- critical phoneme evidence contradicts the expected answer

Short function words are stricter. The layer will not easily accept pairs such as:

- expected `the`, raw `a`
- expected `in`, raw `on`

## Limitations

- This phase uses the greedy Wav2Vec2 transcript only.
- It does not add CTC beam search.
- It does not constrain decoding to content-bank vocabulary.
- It does not train or update any model weights.
- Word alignment is heuristic and intended as metadata/evidence, not as a full forced-alignment engine.

## Future Roadmap

### Future Phase 2

System-wide audit and enforcement to ensure all diagnostic, module, mastery, final assessment, and passage flows use the same shared correction contract.

### Future Phase 3

CTC beam search top-N candidates from Wav2Vec2 logits.

Beam candidates will become extra evidence for the dynamic correction layer, but `raw_transcript` will remain the greedy transcript.

### Future Phase 4

Content-bank and vocabulary-constrained rescoring.

The system will use module CSVs, reading passages, and activity vocabulary as candidate sources.

### Future Phase 5

Correction analytics dashboard and threshold tuning tools for developers.
