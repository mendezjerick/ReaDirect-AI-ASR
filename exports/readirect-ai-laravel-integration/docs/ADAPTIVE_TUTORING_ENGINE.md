# Adaptive Tutoring Engine

The adaptive tutoring engine recommends the next practice item or learning action from recent learner attempts and enriched content metadata. It is a heuristic recommendation system, not an official scorer.

Laravel remains responsible for official progress, module rules, and whether the recommendation is accepted.

## Architecture

Core modules:

- `readirect_asr.adaptive.learner_state`: summarizes recent attempts.
- `readirect_asr.adaptive.remediation_policy`: maps errors to remediation focus.
- `readirect_asr.adaptive.difficulty_policy`: decides increase, same, or decrease.
- `readirect_asr.adaptive.item_selector`: scores and ranks candidates.
- `readirect_asr.adaptive.recommendation`: orchestrates loading candidates and building responses.
- `readirect_asr.adaptive.explanation`: produces safe explanation strings.

Content sources are checked in this order:

1. `content_bank_enriched/enriched_content_index.csv`
2. `data/manifests/content_index.csv`
3. `candidate_items` supplied in the request

## Learner History Schema

Each history item may include `prompt_id`, `module_key`, `activity_type`, `expected_text`, `actual_text`, `is_correct`, `similarity_label`, `error_type`, `skill_signal`, `target_phoneme`, `target_position`, `difficulty_level`, `difficulty_score`, and `timestamp`.

Recent history matters most. The last five attempts influence the immediate next item.

## Candidate Item Schema

Candidate items may include `prompt_id`, `module_key`, `activity_type`, `prompt_text`, `expected_text`, `expected_phonemes`, `phoneme_pattern`, `skill_tag`, `skill_group`, `error_focus`, `target_position`, `target_phoneme`, `difficulty_level`, `difficulty_score`, `adaptive_bucket`, `recommended_for_error_type`, `practice_role`, `mastery_candidate`, `review_candidate`, `is_active`, and `needs_manual_review`.

The engine prefers active items and avoids manual-review items unless there are no better options.

## Policies

Remediation examples:

- `final_sound_error` -> `final_consonant`
- `initial_sound_error` -> `initial_consonant`
- `vowel_error` -> `vowel_sound`
- `skipped_word` -> `sentence_tracking`
- `partial_sentence` -> `fluency_completion`
- `unclear_asr` -> retry recording
- `correct` -> continue or increase

Difficulty rules:

- Correct streak of 3 or recent accuracy >= 0.80 -> increase.
- Incorrect streak of 2 or recent accuracy < 0.50 -> decrease.
- No history -> easy or baseline item.
- Unclear ASR should usually trigger retry before lowering ability estimates.

Difficulty order:

```text
very_easy < easy < medium < hard < very_hard
```

## Item Scoring

Candidates receive positive weight for matching remediation focus, skill signal, target phoneme, target position, target difficulty, active status, same module/activity context, and not being recently used.

Candidates are penalized for being inactive, needing manual review, being recently used, or being a mastery item outside a mastery context.

## API

Endpoint:

```text
POST /recommend-next
```

Example request:

```json
{
  "learner_history": [
    {
      "prompt_id": "M2-001",
      "expected_text": "cat",
      "actual_text": "cap",
      "is_correct": false,
      "error_type": "final_sound_error",
      "skill_signal": "final_consonant",
      "target_phoneme": "T",
      "difficulty_level": "easy"
    }
  ],
  "candidate_items": [
    {
      "prompt_id": "M2-014",
      "module_key": "module_2",
      "activity_type": "display_word_reading",
      "prompt_text": "Read the word.",
      "expected_text": "hat",
      "error_focus": "final_consonant",
      "target_phoneme": "T",
      "difficulty_level": "easy",
      "is_active": true,
      "needs_manual_review": false
    }
  ],
  "top_k": 5,
  "debug": true
}
```

Example response fields:

- `selected_item`
- `ranked_candidates`
- `learner_summary`
- `recommendation`
- `explanation`
- `warnings`

## Laravel Use

Laravel should send recent learner attempt history, current module/activity context, and optional candidate items from the database.

The AI service returns a recommended next item, reason codes, a learner-safe summary, and teacher/developer explanations.

Laravel remains responsible for deciding whether to follow the recommendation, enforcing module rules, saving official progress, and keeping debug fields away from students.

The adaptive engine is advisory. It recommends practice focus and candidate items from AI analysis signals, enriched content metadata, and learner history, but Laravel enforces official eligibility and progression rules.

## Limitations

- Recommendations are heuristic and should be reviewed with educator input.
- ASR mistakes can affect history labels if Laravel uses ASR-derived analysis.
- Content enrichment quality depends on CMUdict coverage and manually reviewed CSV metadata.
- This engine does not train a learner model yet.
