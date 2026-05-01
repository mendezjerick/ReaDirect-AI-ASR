from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


def _normalize(expected: str, raw: str, observed: list[str] | None = None):
    return normalize_asr_transcript(raw_transcript=raw, expected_text=expected, observed_phonemes=observed or [])


def test_letter_l_accepts_bare_blank_and_alias_with_phoneme_evidence() -> None:
    bare = _normalize("L", "l", ["EH", "L"])
    blank = _normalize("L", "", ["EH", "L"])
    alias = _normalize("L", "elle")

    for result in (bare, blank, alias):
        assert result.accepted is True
        assert result.corrected_transcript == "L"
        assert result.displayed_transcript == "L"
        assert result.prompt_type == "letter"

    assert blank.accepted_by_phoneme_evidence is True


def test_letter_z_and_c_wav2vec2_short_outputs_accept_expected_centric() -> None:
    z = _normalize("Z", "zy", ["Z", "IY"])
    c = _normalize("C", "s", ["S", "IY"])

    assert z.accepted is True
    assert z.corrected_transcript == "Z"
    assert c.accepted is True
    assert c.corrected_transcript == "C"


def test_q_requires_initial_k_when_phoneme_evidence_is_available() -> None:
    missing_k = _normalize("Q", "you", ["Y", "UW"])
    with_k = _normalize("Q", "you", ["K", "Y", "UW"])

    assert missing_k.accepted is False
    assert missing_k.displayed_transcript == "you"
    assert missing_k.critical_phoneme == "K"
    assert missing_k.critical_phoneme_detected is False

    assert with_k.accepted is True
    assert with_k.displayed_transcript == "Q"
    assert with_k.critical_phoneme_detected is True


def test_word_known_confusions_are_expected_centric() -> None:
    for expected, raw in [("ten", "then"), ("red", "read"), ("tree", "three")]:
        result = _normalize(expected, raw)
        assert result.accepted is True
        assert result.corrected_transcript == expected
        assert result.displayed_transcript == expected
        assert result.accepted_by_known_confusion is True


def test_negative_letter_and_word_keep_raw_display() -> None:
    letter = _normalize("Z", "banana")
    word = _normalize("ten", "banana")

    assert letter.accepted is False
    assert letter.displayed_transcript == "banana"
    assert word.accepted is False
    assert word.displayed_transcript == "banana"


def test_sentence_uses_raw_wav2vec2_transcript_not_expected_replacement() -> None:
    result = _normalize("I see a tree.", "I see a three.")

    assert result.prompt_type == "sentence"
    assert result.corrected_transcript == "I see a three."
    assert result.displayed_transcript == "I see a three."
    assert result.corrected_transcript != "I see a tree."
    assert result.raw_wer > 0.0
