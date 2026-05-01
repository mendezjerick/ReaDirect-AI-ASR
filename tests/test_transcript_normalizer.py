from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


def _normalize(expected: str, raw: str):
    return normalize_asr_transcript(raw_transcript=raw, expected_text=expected)


def test_known_confusion_and_homophone_expected_prompt_corrections() -> None:
    cases = [
        ("Red", "Read", "Red"),
        ("Read", "Red", "Read"),
        ("Tree", "Three", "Tree"),
        ("Three", "Tree", "Three"),
        ("Right", "Write", "Right"),
        ("Hear", "Here", "Hear"),
    ]

    for expected, raw, corrected in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == corrected
        assert result.displayed_transcript == corrected
        assert result.normalization_applied is True
        assert result.correction_strategy_used == "known_confusion_expected_prompt_alignment"
        assert result.accepted_by_phonetic_threshold is True
        assert result.threshold_used == 0.82
        assert result.raw_wer == 1.0
        assert result.corrected_wer == 0.0


def test_single_letter_spoken_forms_are_corrected_to_expected_letter() -> None:
    cases = [
        ("Z", "zee", "Z"),
        ("Z", "zed", "Z"),
        ("X", "ex", "X"),
        ("X", "axe", "X"),
        ("G", "gee", "G"),
        ("G", "jee", "G"),
        ("C", "see", "C"),
        ("Y", "why", "Y"),
        ("U", "you", "U"),
        ("B", "bee", "B"),
        ("Q", "queue", "Q"),
        ("W", "double you", "W"),
    ]

    for expected, raw, corrected in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == corrected
        assert result.displayed_transcript == corrected
        assert result.normalization_applied is True
        assert result.corrected_wer == 0.0


def test_single_letter_asr_confusions_use_expected_prompt_threshold() -> None:
    cases = [
        ("Z", "they", "Z", 0.90),
        ("Z", "the", "Z", 0.86),
        ("Z", "see", "Z", 0.86),
        ("Z", "c", "Z", 0.85),
    ]

    for expected, raw, corrected, score in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == corrected
        assert result.displayed_transcript == corrected
        assert result.phonetic_similarity_score == score
        assert result.normalization_applied is True
        assert result.accepted_by_phonetic_threshold is True
        assert result.threshold_used == 0.85
        assert result.correction_strategy_used == "letter_phonetic_threshold_alignment"


def test_unrelated_transcripts_are_not_corrected_or_display_replaced() -> None:
    cases = [
        ("Z", "banana"),
        ("Red", "table"),
        ("Tree", "car"),
        ("G", "apple"),
        ("X", "dog"),
    ]

    for expected, raw in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == raw
        assert result.displayed_transcript == raw
        assert result.normalization_applied is False
        assert result.accepted_by_phonetic_threshold is False
        assert result.correction_strategy_used == "none"
        assert result.corrected_wer == result.raw_wer


def test_displayed_transcript_uses_expected_text_only_when_accepted() -> None:
    accepted = [
        ("Z", "They", "Z"),
        ("X", "ex", "X"),
        ("Red", "Read", "Red"),
        ("Tree", "Three", "Tree"),
    ]

    for expected, raw, displayed in accepted:
        result = _normalize(expected, raw)
        assert result.displayed_transcript == displayed

    rejected = _normalize("Z", "Banana")
    assert rejected.corrected_transcript == "Banana"
    assert rejected.displayed_transcript == "Banana"
