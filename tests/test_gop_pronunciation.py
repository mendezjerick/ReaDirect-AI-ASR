from readirect_asr.pronunciation.gop import apply_gop_to_transcript_meta, compute_gop
from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


def _normalize(raw: str, expected: str, prompt_type: str, observed: list[str]):
    return normalize_asr_transcript(
        raw_transcript=raw,
        expected_text=expected,
        prompt_type=prompt_type,
        observed_phonemes=observed,
    )


def test_word_accepted_by_gop_pronunciation_evidence() -> None:
    normalization = _normalize("Layo", "Leo", "word", ["L", "EY", "OW"])
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="Leo",
        prompt_type="word",
        raw_transcript="Layo",
        observed_phonemes=["L", "EY", "OW"],
        config={"word_threshold": 0.75},
    )
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert gop["gop_score"] >= 0.75
    assert updated["accepted"] is True
    assert updated["corrected_transcript"] == "Leo"
    assert updated["displayed_transcript"] == "Leo"
    assert updated["correction_strategy_used"] == "gop_pronunciation_evidence"
    assert updated["gop_correction_applied"] is True


def test_word_rejected_by_low_gop_does_not_force_expected_text() -> None:
    normalization = _normalize("banana", "Leo", "word", ["B", "AH", "N", "AE", "N", "AH"])
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="Leo",
        prompt_type="word",
        raw_transcript="banana",
        observed_phonemes=["B", "AH", "N", "AE", "N", "AH"],
        config={"word_threshold": 0.75},
    )
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert gop["gop_decision"] == "rejected_low_gop"
    assert updated["accepted"] is False
    assert updated["displayed_transcript"] != "Leo"


def test_letter_accepted_by_gop_pronunciation_evidence() -> None:
    normalization = _normalize("See", "C", "letter", ["S", "IY"])
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="C",
        prompt_type="letter",
        raw_transcript="See",
        observed_phonemes=["S", "IY"],
        config={"letter_threshold": 0.70},
    )
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert gop["gop_decision"] == "accepted_by_pronunciation_evidence"
    assert updated["accepted"] is True
    assert updated["corrected_transcript"] == "C"
    assert updated["displayed_transcript"] == "C"


def test_gop_skips_bad_audio() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="Leo",
        prompt_type="word",
        raw_transcript="Layo",
        observed_phonemes=["L", "EY", "OW"],
        retry_required=True,
    )

    assert gop["gop_available"] is False
    assert gop["gop_decision"] == "skipped_bad_audio"


def test_gop_skips_missing_expected_text() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="",
        prompt_type="word",
        raw_transcript="Layo",
        observed_phonemes=["L", "EY", "OW"],
    )

    assert gop["gop_available"] is False
    assert gop["gop_decision"] == "skipped_no_expected_text"


def test_sentence_gop_does_not_force_full_expected_sentence() -> None:
    normalization = _normalize("Layo can read", "Leo can read", "sentence", ["L", "EY", "OW", "K", "AE", "N", "R", "IY", "D"])
    gop = {
        "gop_enabled": True,
        "gop_available": True,
        "gop_score": 0.82,
        "gop_confidence": 0.78,
        "gop_decision": "accepted_by_pronunciation_evidence",
        "gop_threshold": 0.70,
        "gop_prompt_type": "sentence",
        "gop_expected_phonemes": ["L", "IY", "OW"],
        "gop_observed_phonemes": ["L", "EY", "OW"],
        "gop_phoneme_scores": [],
        "gop_word_scores": [{"word": "Leo", "score": 0.82, "status": "acceptable"}],
        "mispronounced_phonemes": [],
        "weak_words": [],
        "gop_error": None,
    }
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert updated["displayed_transcript"] != "Leo can read"
    assert updated["displayed_transcript"] == "Layo can read"
    assert updated["gop_word_scores"][0]["word"] == "Leo"
