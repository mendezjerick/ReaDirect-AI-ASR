# Acoustic GOP Pronunciation Evidence

Acoustic GOP in ReaDirect is a Goodness of Pronunciation evidence layer that uses the existing Wav2Vec2 phoneme model. It does not replace the normal ASR transcript, the expected-centric correction algorithm, learner attempt flow, or scoring routes.

The acoustic GOP module uses phoneme-level acoustic evidence from the learner's recorded audio. The expected phoneme sequence is aligned with frame-level phoneme probabilities or logits produced by the phoneme model, and each phoneme is scored based on how strongly the audio supports the expected sound. This provides additional evidence for identifying likely vowel confusion, consonant substitution, and omitted sounds. The system remains a reading-support and decision-support tool and does not replace teacher judgment.

## Flow

1. Receive learner audio and expected item metadata.
2. Preprocess audio through the existing ASR provider path.
3. Generate the normal Wav2Vec2 ASR transcript.
4. Run the existing `models/wav2vec2-phoneme` model and collect frame-level log probabilities.
5. Convert the expected item into canonical phonemes.
6. CTC-align the expected phonemes against frame-level phoneme probabilities.
7. Compute per-phoneme acoustic GOP scores.
8. Pass GOP evidence into the expected-centric decision layer.
9. Return and store GOP fields with the normal ASR result.

## How This Differs From Transcript Phoneme Comparison

Transcript-based phoneme comparison checks whether the expected answer and ASR transcript have similar phoneme sequences. Acoustic GOP scores the learner audio itself. For each expected phoneme, it compares the model's acoustic support for that phoneme against the strongest competing phoneme during the aligned frames:

```text
GOP(phone) = average(log P(expected_phone | frame) - max log P(other_phone | frame))
```

The margin is normalized to a `0.0..1.0` score for API and UI display.

## Letter Tasks

Letter tasks must pass the correct task metadata:

- `letter_sound`: printed `B` maps to the sound phoneme `/B/`.
- `letter_name`: printed `B` maps to the spoken letter-name sequence.
- `word`, `sentence`, and `passage`: expected text maps through the existing phoneme dictionary/mapping path.

## Configuration

Relevant environment values:

```text
ENABLE_ACOUSTIC_GOP=true
GOP_MODEL_PATH=models/wav2vec2-phoneme
GOP_MODEL_NAME=existing_wavtec_phoneme_model
GOP_MIN_ALIGNMENT_QUALITY=0.25
GOP_WEAK_THRESHOLD=0.55
GOP_ACCEPTABLE_THRESHOLD=0.75
GOP_DEBUG=false
```

`GOP_ENABLED` remains accepted as a backward-compatible alias for `ENABLE_ACOUSTIC_GOP`.

When `ENABLE_ACOUSTIC_GOP=false`, the service returns GOP as disabled and the existing expected-centric ASR behavior is unchanged.

## API Fields

ASR responses may include:

- `gop_enabled`
- `gop_supported`
- `gop_model_version`
- `gop_model_path`
- `gop_score`
- `overall_gop_score`
- `acoustic_confidence`
- `canonical_phonemes`
- `canonical_expected_phonemes`
- `decoded_phonemes`
- `decoded_acoustic_phonemes`
- `phoneme_scores`
- `weak_phoneme`
- `weak_phoneme_score`
- `lowest_phoneme`
- `lowest_phoneme_score`
- `nearest_confusion`
- `alignment_quality`
- `gop_frame_count`
- `gop_duration_seconds`
- `gop_fallback_used`
- `gop_error`

Existing records without these fields remain valid.

## Fallback Behavior

GOP failures must not block learner attempts. If audio is invalid, alignment fails, a phoneme token is missing, the phoneme model is unavailable, or GOP is disabled, the response reports `gop_supported=false` or `alignment_quality=failed` and the expected-centric decision layer continues with the transcript and phoneme comparison evidence it already used.

GOP failure alone must not mark a learner wrong.

## Expected-Centric Integration

The expected-centric layer remains the final decision layer. Acoustic GOP can enrich technical classifications such as:

- `vowel_confusion`
- `consonant_confusion`
- `initial_sound_substitution`
- `medial_sound_confusion`
- `final_sound_omission`
- `phoneme_omission`
- `phoneme_insertion`
- `low_confidence_audio`
- `correct_with_low_confidence`
- `correct`

GOP supports these classifications with fields like `weak_phoneme`, `nearest_confusion`, `lowest_phoneme_score`, and `alignment_quality`. It is not the only basis for correctness.

## True Sandbox Debug

The Laravel True Sandbox displays acoustic GOP values when available:

- enabled and supported flags
- model/version
- alignment quality
- overall GOP
- weak phoneme and weak score
- nearest acoustic competitor
- expected and decoded acoustic phonemes
- per-phoneme score rows
- GOP error and fallback state

This is admin/debug visibility only.

## AI Banner Status

The admin AI status banner shows a compact GOP status:

- `GOP: Off`: acoustic GOP is disabled.
- `GOP: Ready`: GOP is enabled and the phoneme model is available.
- `GOP: Active`: a result reports usable GOP alignment.
- `GOP: Fallback`: GOP was attempted but the system used expected-centric fallback.
- `GOP: Failed`: GOP is enabled but the service/model is not available or alignment failed.

## Local Debug Script

Use the debug script to inspect one recording:

```bash
python scripts/debug_acoustic_gop.py storage/audio/example.wav --expected log --prompt-type word --task-type word --transcript lug
```

It prints expected phonemes, decoded acoustic phonemes, alignment quality, per-phoneme GOP scores, the weakest phoneme, nearest competitor, and the expected-centric decision support fields.

## Tests

The unit tests use synthetic frame-level log probabilities so they can run without sample learner audio:

```bash
python -m pytest tests/test_gop_pronunciation.py
```

Covered cases include correct high-GOP evidence, vowel confusion, consonant confusion, final sound omission, invalid audio fallback, missing phoneme mapping, GOP disabled fallback, alignment failure fallback, and letter sound/name mapping.

## Limitations

The current GOP values still require calibration against real learner recordings. CTC alignment is practical and fast, but it is approximate. Score thresholds should be validated with real classroom audio before being treated as high-confidence placement or mastery evidence.

Deterministic Miss Ciel feedback and any LLM feedback layer are intentionally not part of this implementation.
