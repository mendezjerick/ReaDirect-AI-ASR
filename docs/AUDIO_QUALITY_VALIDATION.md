# Audio Quality Validation

The AI service now runs lightweight audio quality analysis around the Wav2Vec2-only ASR path. This is runtime validation and metadata generation only. It does not train models, modify model weights, or reintroduce Whisper.

## What It Detects

- Minimum duration: recordings shorter than `AUDIO_MIN_DURATION_SECONDS`.
- Silent or mostly silent audio: energy-based speech ratio below configured thresholds.
- Low volume: RMS dBFS below `AUDIO_LOW_VOLUME_DBFS`.
- Clipping/distortion: samples at or above `AUDIO_CLIPPING_THRESHOLD` exceeding `AUDIO_CLIPPED_RATIO_THRESHOLD`.
- Speech segments: simple frame-level RMS detection using `AUDIO_SILENCE_DBFS`.
- Pauses: silence gaps between detected speech segments, including long and very long pause counts.
- ASR uncertainty: blank transcript, missing phoneme evidence, low confidence, or bad audio flags.

## Runtime Flow

```text
audio file
-> audio quality validation
-> optional strict quality gate
-> Wav2Vec2 ASR
-> phoneme evidence and transcript correction
-> uncertainty / retry-required decision
-> API response metadata
```

If strict quality gating is enabled and audio is too short, silent, mostly silent, too quiet, or clipped, ASR is skipped and the response is marked `retry_required=true`, `uncertain=true`, and `accepted=false`. This protects learners from being scored wrong for unusable recordings.

If quality warnings are non-critical or the audio cannot be inspected but the file path is otherwise acceptable, the service preserves existing behavior and continues to ASR while returning warnings.

## Configuration

Defaults are in `configs/service_config.yaml` and `.env.example`:

```text
AUDIO_MIN_DURATION_SECONDS=1.0
AUDIO_MAX_DURATION_SECONDS=30.0
AUDIO_LOW_VOLUME_DBFS=-35.0
AUDIO_SILENCE_DBFS=-40.0
AUDIO_MOSTLY_SILENT_RATIO=0.85
AUDIO_CLIPPING_THRESHOLD=0.98
AUDIO_CLIPPED_RATIO_THRESHOLD=0.01
AUDIO_MIN_SPEECH_RATIO=0.15
AUDIO_LONG_PAUSE_SECONDS=1.0
AUDIO_VERY_LONG_PAUSE_SECONDS=2.0
AUDIO_ENABLE_QUALITY_GATE=true
AUDIO_RETRY_ON_BAD_QUALITY=true
```

## API Fields

`POST /analyze-audio` returns:

- `audio_quality`
- `pause_metrics`
- `uncertain`
- `retry_required`
- `uncertainty_reasons`
- `quality_gate_failed`
- `learner_retry_message`
- `developer_quality_notes`

When `retry_required=true`, the service does not mark the learner correct and does not force `expected_text` into `displayed_transcript`.

## Health Metadata

`GET /health` reports:

- `audio_quality_validation_enabled`
- `audio_quality_thresholds`
- `pause_detection_enabled`
- `uncertainty_decision_enabled`
