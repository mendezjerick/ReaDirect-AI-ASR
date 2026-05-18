# Passage-Aware Dynamic Alignment Repair

Phase 1B adds split/merge aware alignment for sentence, paragraph, and passage prompts.

## Why Phase 1 Was Not Enough

Phase 1 dynamic expected-word correction worked after a word pair had already been aligned. That helped clean one-to-one cases:

- `shield` / `shild`
- `knights` / `nights`
- `Leo` / `Layo`

Passage ASR often fails at word boundaries instead:

- `time after` / `timeafter`
- `open Maya` / `openmya`
- `fruit seeds` / `fruitsieds`
- `nearby` / `near by`

When literal word alignment pairs the wrong tokens, correction happens too late. Phase 1B moves correction evidence into the alignment process itself.

## Alignment Repair

The passage aligner uses dynamic programming and scores variable-size chunks:

- 1 expected word to 1 raw word
- 1 expected word to 2 raw words
- 2 expected words to 1 raw word
- 2 expected words to 2 raw words
- restricted 3 to 1 and 1 to 3 repairs for obvious boundary errors

Each candidate compares:

- spaced text, such as `time after`
- joined text, such as `timeafter`
- vowel-tolerant similarity
- consonant skeleton similarity
- phrase phoneme similarity when available
- GOP word scores when available
- expected-position context

## Statuses

`word_alignment` can now return:

- `exact_correct`
- `accepted_by_dynamic_expected_word_correction`
- `accepted_by_homophone`
- `accepted_by_phoneme_similarity`
- `accepted_by_gop`
- `accepted_by_asr_spelling_variant`
- `accepted_by_split_merge`
- `partial`
- `incorrect`
- `missing`
- `inserted`

Accepted statuses count as correct for passage highlighting and incorrect-word counts. `partial` remains a caution state and is not counted as fully correct.

## Transcript Contract

Phase 1B does not hide bad ASR.

- `raw_transcript` remains direct Wav2Vec2 output.
- Full passage `displayed_transcript` remains ASR-based.
- The expected passage is not forced into `displayed_transcript`.
- Repairs are stored per word/chunk inside `word_alignment`.
- Admin/debug metadata includes chunk operation, confidence, similarities, and reason.

Example:

```json
[
  {
    "expected_word": "time",
    "recognized_word": "timeafter",
    "expected_chunk": "time after",
    "recognized_chunk": "timeafter",
    "operation": "merge_match",
    "status": "accepted_by_split_merge",
    "counts_as_correct": true
  },
  {
    "expected_word": "after",
    "recognized_word": "timeafter",
    "expected_chunk": "time after",
    "recognized_chunk": "timeafter",
    "operation": "merge_match",
    "status": "accepted_by_split_merge",
    "counts_as_correct": true
  }
]
```

## Thresholds

Defaults:

- `DYNAMIC_ALIGNMENT_REPAIR_ENABLED=true`
- `DYNAMIC_ALIGNMENT_ALLOW_SPLIT_MERGE=true`
- `DYNAMIC_ALIGNMENT_MAX_EXPECTED_CHUNK=3`
- `DYNAMIC_ALIGNMENT_MAX_RAW_CHUNK=3`
- `DYNAMIC_ALIGNMENT_ACCEPT_THRESHOLD=0.82`
- `DYNAMIC_ALIGNMENT_PARTIAL_THRESHOLD=0.65`
- `DYNAMIC_ALIGNMENT_SPLIT_MERGE_THRESHOLD=0.84`
- `DYNAMIC_ALIGNMENT_HOMOPHONE_THRESHOLD=0.96`
- `DYNAMIC_ALIGNMENT_GOP_ACCEPT_THRESHOLD=0.75`
- `DYNAMIC_ALIGNMENT_SHORT_FUNCTION_WORD_STRICT=true`
- `DYNAMIC_ALIGNMENT_DEBUG=true`

Passage thresholds are intentionally stricter than isolated word thresholds.

## Safety Rules

The aligner does not accept:

- retry-required audio
- bad audio based on upstream gates
- risky short function word substitutions like `the` / `a`
- unrelated words with weak spelling and weak phoneme/GOP evidence
- ambiguous 3-token repairs unless confidence is very high

Examples:

- `nearby` / `near by`: accepted by split/merge.
- `fruit seeds` / `fruitsieds`: accepted only when boundary and pronunciation-like evidence is strong enough.
- `woven` / `woman`: incorrect or partial unless stronger acoustic evidence is present.
- `the` / `a`: incorrect.

## Laravel Use

Laravel does not compute this repair. It receives `word_alignment` from FastAPI.

Diagnostic passage reading now uses `word_alignment` when present to avoid marking accepted split/merge chunks red and to set the automatic incorrect-word count. Final assessment passage upload also uses the alignment count when available.

## Future Roadmap

Phase 2: System-wide verification that all diagnostic, module, mastery, final assessment, and reassessment reading flows use the same passage-aware alignment service.

Phase 3: CTC beam search top-N candidates from Wav2Vec2 logits as extra evidence for dynamic correction.

Phase 4: Content-bank and vocabulary-constrained rescoring.

Phase 5: Developer analytics dashboard for threshold tuning and correction audit.

Beam search, constrained decoding, model training, LLMs, and Whisper are not part of Phase 1B.
