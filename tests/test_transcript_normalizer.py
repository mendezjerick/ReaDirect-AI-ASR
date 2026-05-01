from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


def _normalize(expected: str, raw: str):
    return normalize_asr_transcript(raw_transcript=raw, expected_text=expected)


def test_known_confusion_and_homophone_expected_prompt_corrections() -> None:
    cases = [
        ("Ten", "Then", "Ten"),
        ("Then", "Ten", "Then"),
        ("Thin", "Tin", "Thin"),
        ("Tin", "Thin", "Tin"),
        ("Red", "Read", "Red"),
        ("Read", "Red", "Read"),
        ("Tree", "Three", "Tree"),
        ("Three", "Tree", "Three"),
        ("See", "Sea", "See"),
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
        assert result.accepted_by_known_confusion is True
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
        assert result.accepted_by_letter_normalization is True
        assert result.corrected_wer == 0.0


def test_single_letter_asr_confusions_use_expected_prompt_threshold() -> None:
    cases = [
        ("D", "they", "D", 0.86),
        ("V", "they", "V", 0.86),
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


def test_generated_letter_lattice_variants_are_corrected_to_expected_letter() -> None:
    cases = [
        ("Z", "ZY", True),
        ("Z", "Zi", True),
        ("Z", "Zii", True),
        ("Z", "Zih", True),
        ("Z", "Zee", False),
        ("Z", "Zey", True),
        ("B", "Bi", True),
        ("C", "Cy", True),
        ("D", "Dy", True),
        ("G", "Gy", True),
        ("T", "Ty", True),
        ("V", "Vy", True),
    ]

    for expected, raw, should_use_lattice in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == expected
        assert result.displayed_transcript == expected
        assert result.normalization_applied is True
        assert result.accepted_by_letter_lattice is should_use_lattice
        assert result.accepted_by_phonetic_threshold is should_use_lattice
        assert result.phonetic_similarity_score >= 0.85
        assert result.threshold_used == 0.85
        assert result.correction_strategy_used in {
            "expected_centric_phonetic_lattice_matching",
            "letter_pronunciation_normalization",
        }


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
        assert result.accepted_by_known_confusion is False
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


def test_positive_word_level_corrections_display_expected_csv_answer() -> None:
    cases = [
        ("ten", "then"),
        ("then", "ten"),
        ("thin", "tin"),
        ("tin", "thin"),
        ("red", "read"),
        ("tree", "three"),
        ("see", "sea"),
        ("right", "write"),
        ("hear", "here"),
    ]

    for expected, raw in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == expected
        assert result.displayed_transcript == expected
        assert result.accepted_by_phonetic_threshold is True
        assert result.accepted_by_known_confusion is True
        assert result.corrected_wer == 0.0


def test_negative_word_level_transcripts_keep_recognized_display() -> None:
    cases = [
        ("ten", "banana"),
        ("red", "table"),
        ("tree", "car"),
        ("Z", "banana"),
    ]

    for expected, raw in cases:
        result = _normalize(expected, raw)

        assert result.corrected_transcript == raw
        assert result.displayed_transcript == raw
        assert result.normalization_applied is False
        assert result.accepted_by_phonetic_threshold is False


def test_sentence_prompts_do_not_use_word_level_expected_replacement() -> None:
    unrelated = _normalize("The red hen is in the pen.", "unrelated sentence")
    near_match = _normalize("I see a tree.", "I see a three.")

    assert unrelated.corrected_transcript == "unrelated sentence"
    assert unrelated.displayed_transcript == "unrelated sentence"
    assert unrelated.correction_strategy_used == "none"

    assert near_match.corrected_transcript == "I see a three."
    assert near_match.displayed_transcript == "I see a three."
    assert near_match.normalization_applied is False
