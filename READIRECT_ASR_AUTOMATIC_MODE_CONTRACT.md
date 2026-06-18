# ReaDirect-AI-ASR Automatic Mode Contract Readiness

Date prepared: 2026-06-17

Scope: documentation only. No ASR endpoint behavior, field names, thresholds, GOP behavior, model loading, or transcript correction logic was changed.

Prompt 3 implementation note: Laravel Automatic Ciel Listening Mode now submits finalized speech chunks through the existing Laravel audio upload and ASR orchestration path. ReaDirect-AI-ASR code was not changed in Prompt 3, and `/analyze-audio` remains speech-analysis only.

## Current Endpoint Status

`POST /analyze-audio` is already sufficient for a future Laravel-owned Automatic Ciel Listening Mode because it accepts an audio file path plus expected-text context and returns transcript, quality, retry, correction, error, and pronunciation metadata.

Laravel should continue to call ASR through its existing `AIAnalysisResolver` and `ReadirectAIService` path. ASR remains speech-analysis only and must not own official scoring, learner progression, placement, mastery, or Ciel behavior.

## Request Fields Already Available

Declared `AnalyzeAudioRequest` fields:

- `audio_path`
- `expected_text`
- `accepted_answers`
- `prompt_id`
- `prompt_type`
- `module_key`
- `activity_type`
- `task_type`
- `learner_response_id`
- `attempt_id`
- `content_metadata`
- `learner_history`
- `candidate_items`
- `debug`
- `developer_reinforcement_enabled`
- `developer_user_role`
- `developer_user_id`

Prompt 3 should keep automatic-listening session/chunk metadata in Laravel unless an ASR-specific need appears. Unknown extra fields are not required for the current ASR contract.

## Response Fields Useful For Prompt 3

Existing fields relevant to automatic mode:

- `transcript`
- `raw_transcript`
- `corrected_transcript`
- `displayed_transcript`
- `expected_text`
- `accepted`
- `is_correct`
- `is_accepted`
- `confidence`
- `confidence_level`
- `raw_wer`
- `corrected_wer`
- `raw_cer`
- `corrected_cer`
- `audio_quality`
- `pause_metrics`
- `retry_required`
- `uncertain`
- `uncertainty_reasons`
- `quality_gate_failed`
- `learner_retry_message`
- `error_type`
- `feedback_hint`
- `target_phoneme`
- `target_position`
- `gop_enabled`
- `gop_available`
- `gop_supported`
- `gop_score`
- `overall_gop_score`
- `gop_confidence`
- `acoustic_confidence`
- `gop_decision`
- `gop_phoneme_scores`
- `phoneme_scores`
- `mispronounced_phonemes`
- `weak_phoneme`
- `lowest_phoneme`
- `observed_phonemes`
- `actual_phonemes`
- `phoneme_similarity`
- `word_alignment`
- `processing_seconds`
- `warnings`
- `error`

GOP/phoneme fields are conditional and must stay optional in Laravel. Runtime model availability determines whether they contain scores or not-available metadata.

## Current Bad-Audio Outcomes

The current audio quality path can report:

- `audio_too_short`
- `audio_too_long`
- `mostly_silent`
- `low_volume`
- `clipped`
- `no_speech_detected`
- `audio_quality_unreadable`
- `quality_gate_failed=true`
- `retry_required=true`
- `uncertain=true`
- `learner_retry_message`
- `developer_quality_notes`

Prompt 3 should treat these as retry/teaching inputs after Laravel receives the ASR result. ASR thresholds were not changed in Prompt 2.

## Non-Goals For Prompt 2

- No live VAD.
- No streaming ASR.
- No chunk/session id handling in ASR.
- No WPM/WCPM fields added to ASR. Laravel currently computes WPM/WCPM where needed.
- No forced GOP when runtime phoneme models are unavailable.
- No response field renames or removals.

## Prompt 3 Guidance

Future automatic mode should send completed speech chunks through Laravel's existing audio upload and ASR orchestration. Laravel should own duplicate prevention, session/chunk ids, official scoring, Ciel calls, and learner-facing state transitions.
