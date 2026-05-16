from pathlib import Path

from readirect_asr.correction.dynamic_expected_word_correction import (
    apply_dynamic_expected_word_correction,
    correct_expected_word,
    dynamic_word_alignment,
)
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _loader(tmp_path: Path) -> CMUDictLoader:
    cmu = tmp_path / "cmu"
    cmu.mkdir(exist_ok=True)
    (cmu / "cmudict.dict").write_text(
        "\n".join(
            [
                "KNIGHTS N AY1 T S",
                "NIGHTS N AY1 T S",
                "PULLED P UH1 L D",
                "POLED P OW1 L D",
                "THE DH AH0",
                "A AH0",
            ]
        ),
        encoding="utf-8",
    )
    (cmu / "cmudict.phones").write_text("N nasal\nAY vowel\nT stop\nS fricative\nP stop\nUH vowel\nL liquid\nD stop\nOW vowel\nDH fricative\nAH vowel\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("N\nAY\nAY1\nT\nS\nP\nUH\nUH1\nL\nD\nOW\nOW1\nDH\nAH\nAH0\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def test_letters_accept_spoken_forms() -> None:
    for expected, raw in [("C", "See"), ("L", "Elle"), ("Q", "You")]:
        result = correct_expected_word(expected, raw, "letter")

        assert result["accepted"] is True
        assert result["corrected_text"] == expected
        assert result["display_text"] == expected


def test_word_accepts_gop_supported_near_match_and_rejects_unrelated(tmp_path: Path) -> None:
    accepted = correct_expected_word("Leo", "Layo", "word", gop_score=0.84, cmudict_loader=_loader(tmp_path))
    rejected = correct_expected_word("Leo", "banana", "word", cmudict_loader=_loader(tmp_path))

    assert accepted["accepted"] is True
    assert accepted["corrected_text"] == "Leo"
    assert accepted["sub_strategy"] == "gop_supported_expected_match"
    assert rejected["accepted"] is False
    assert rejected["display_text"] == "banana"


def test_word_accepts_dynamic_spelling_and_homophone_matches(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    spelling = correct_expected_word("shield", "shild", "word", cmudict_loader=loader)
    homophone = correct_expected_word("knights", "nights", "word", cmudict_loader=loader)

    assert spelling["accepted"] is True
    assert spelling["sub_strategy"] == "spelling_context_expected_match"
    assert homophone["accepted"] is True
    assert homophone["homophone_match"] is True


def test_dynamic_safety_rejects_weak_or_short_function_word_matches(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    assert correct_expected_word("pulled", "poled", "passage", cmudict_loader=loader)["accepted"] is False
    assert correct_expected_word("the", "a", "passage", cmudict_loader=loader)["accepted"] is False


def test_sentence_word_alignment_marks_dynamic_and_homophone_acceptance(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    shield = dynamic_word_alignment("Arthur carried his shield", "arthur carried his shild", "passage", cmudict_loader=loader)
    knights = dynamic_word_alignment("Many knights pulled", "many nights pulled", "passage", cmudict_loader=loader)

    assert shield[-1]["status"] == "accepted_by_dynamic_expected_word_correction"
    assert shield[-1]["counts_as_correct"] is True
    assert knights[1]["status"] == "accepted_by_homophone"
    assert knights[1]["counts_as_correct"] is True


def test_sentence_metadata_does_not_force_full_displayed_transcript(tmp_path: Path) -> None:
    meta = {
        "expected_text": "Arthur carried his shield",
        "raw_transcript": "arthur carried his shild",
        "corrected_transcript": "arthur carried his shild",
        "displayed_transcript": "arthur carried his shild",
        "prompt_type": "passage",
        "debug_metadata": {},
    }

    updated = apply_dynamic_expected_word_correction(meta, cmudict_loader=_loader(tmp_path))

    assert updated["raw_transcript"] == "arthur carried his shild"
    assert updated["corrected_transcript"] == "arthur carried his shild"
    assert updated["displayed_transcript"] == "arthur carried his shild"
    assert updated["word_alignment"][-1]["status"] == "accepted_by_dynamic_expected_word_correction"


def test_dynamic_correction_skips_retry_uncertain_and_missing_expected() -> None:
    retry = correct_expected_word("shield", "shild", "word", retry_required=True)
    uncertain = correct_expected_word("shield", "shild", "word", uncertain=True)
    missing = correct_expected_word("", "shild", "word")

    assert retry["accepted"] is False
    assert retry["sub_strategy"] == "skipped_retry_required"
    assert uncertain["accepted"] is False
    assert uncertain["sub_strategy"] == "skipped_uncertain_audio"
    assert missing["accepted"] is False
    assert missing["sub_strategy"] == "skipped_no_expected_text"
