# Laravel Integration Contract

Laravel remains the official scorer. The AI API provides analysis signals only.

## What Laravel Sends

For text analysis:

```json
{
  "prompt_id": "M2-001",
  "expected_text": "cat",
  "actual_text": "cap",
  "accepted_answers": ["cat"],
  "debug": false
}
```

For audio analysis:

```json
{
  "prompt_id": "M2-001",
  "audio_path": "/private/path/audio.wav",
  "expected_text": "cat",
  "accepted_answers": ["cat"],
  "learner_response_id": "123",
  "attempt_id": "456",
  "debug": false
}
```

## What AI Returns

Laravel can save:

- `transcript`
- `raw_transcript`
- `wav2vec2_transcript`
- `corrected_transcript`
- `displayed_transcript`
- `normalized_transcript`
- `provider`
- `asr_route`
- `model_family`
- `model_used`
- `prompt_type`
- `model_size`
- `raw_wer`
- `corrected_wer`
- `raw_cer`
- `corrected_cer`
- `composite_score`
- `accepted`
- `is_correct`
- `is_exact`
- `is_accepted`
- `character_similarity`
- `token_similarity`
- `similarity_label`
- `expected_phonemes`
- `observed_phonemes`
- `actual_phonemes`
- `phoneme_similarity`
- `error_type`
- `error_position`
- `feedback_hint`
- `coach_hint_key`
- `learner_safe_summary`
- `skill_signal`
- `target_phoneme`
- `target_position`
- `recommended_practice_focus`
- `recommended_action`
- `audio_quality`
- `pause_metrics`
- `uncertain`
- `retry_required`
- `uncertainty_reasons`
- `quality_gate_failed`
- `learner_retry_message`
- `developer_quality_notes`
- `content_metadata`
- `enrichment_metadata`
- `warnings`
- `error`

`transcript` is the backward-compatible learner-facing transcript and maps to `displayed_transcript`. `raw_transcript` always preserves the direct Wav2Vec2 ASR output. Whisper runtime fields are not live; if `whisper_transcript` is present, it is `null`, and `whisper_removed=true`.

For accepted letter and word prompts, `corrected_transcript` and `displayed_transcript` are the expected answer. For rejected letter and word prompts, they remain the recognized transcript. Sentence prompts are not forced to `expected_text`; they use Wav2Vec2 transcript WER/CER.

When `retry_required=true`, Laravel should prompt the learner to record again instead of treating the response as a wrong answer. The AI service sets `accepted=false`, `is_correct=false`, and avoids forcing `expected_text` into `displayed_transcript` for retry-required audio. `audio_quality` and `pause_metrics` are runtime metadata only; they do not train models or modify Wav2Vec2 weights.

## Fallback Behavior

If the AI API is offline or returns `ok=false`, Laravel should fall back to existing rule-based scoring and store the AI error/warning only for admin review.

## Debug Fields

`debug_info` is returned only when request `debug=true` and API debug is enabled. Do not show raw debug data to students during strict assessment.

## Privacy

Do not send learner names, emails, school identifiers, or private metadata to the AI service. Use anonymized IDs.

## Scoring Reminder

Official scoring remains in Laravel. AI fields support feedback, analysis, and future adaptive practice.

The AI API should be treated as an analysis service, not as the sole scoring authority. ASR output may be imperfect, so Laravel should combine AI signals with rule-based assessment logic and administrative review tools. For letters and words, WER alone is not the final decision because expected-centric phonetic scoring and phoneme evidence are used.

## Recommend Next Contract

Endpoint:

```text
POST /recommend-next
```

Suggested Laravel `.env` value:

```text
READIRECT_AI_RECOMMEND_NEXT_ENDPOINT=/recommend-next
```

Laravel sends recent learner history, optional current module context, and optional candidate items from the Laravel database.

Example:

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
  "current_context": {
    "module_key": "module_2",
    "activity_type": "read_word"
  },
  "candidate_items": [],
  "top_k": 5
}
```

The AI service returns:

- `selected_item`
- `ranked_candidates`
- `learner_summary`
- `recommendation`
- `explanation`
- `warnings`

Laravel may save the recommendation and reason codes for audit/admin review, but Laravel should still enforce official module progression and item eligibility.

## Laravel Environment Variables

```text
READIRECT_AI_ENABLED=true
READIRECT_AI_BASE_URL=http://127.0.0.1:8001
READIRECT_AI_API_TOKEN=
READIRECT_AI_TIMEOUT_SECONDS=60
READIRECT_AI_ANALYZE_AUDIO_ENDPOINT=/analyze-audio
READIRECT_AI_ANALYZE_TEXT_ENDPOINT=/analyze-text
READIRECT_AI_RECOMMEND_NEXT_ENDPOINT=/recommend-next
READIRECT_AI_CONTENT_ITEM_ENDPOINT=/content-item
```

The main Laravel repository does not need Speechocean762, training JSONL files, training manifests, or checkpoints.
