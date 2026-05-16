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
                "FISH F IH1 SH",
                "TREE T R IY1",
                "HAND HH AE1 N D",
                "HUND HH AH1 N D",
                "HEAD HH EH1 D",
            ]
        ),
        encoding="utf-8",
    )
    (cmu / "cmudict.phones").write_text("N nasal\nAY vowel\nT stop\nS fricative\nP stop\nUH vowel\nL liquid\nD stop\nOW vowel\nDH fricative\nAH vowel\nF fricative\nIH vowel\nSH fricative\nR liquid\nIY vowel\nHH aspirate\nAE vowel\nEH vowel\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("N\nAY\nAY1\nT\nS\nP\nUH\nUH1\nL\nD\nOW\nOW1\nDH\nAH\nAH0\nAH1\nF\nIH\nIH1\nSH\nR\nIY\nIY1\nHH\nAE\nAE1\nEH\nEH1\n", encoding="utf-8")
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
    assert spelling["sub_strategy"] == "vowel_tolerant_consonant_skeleton_match"
    assert homophone["accepted"] is True
    assert homophone["homophone_match"] is True


def test_asr_spelling_variant_accepts_vowel_tolerant_expected_word(tmp_path: Path) -> None:
    result = correct_expected_word("hand", "hund", "word", cmudict_loader=_loader(tmp_path))

    assert result["accepted"] is True
    assert result["corrected_text"] == "hand"
    assert result["display_text"] == "hand"
    assert result["strategy"] == "dynamic_asr_spelling_variant"
    assert result["sub_strategy"] == "vowel_tolerant_consonant_skeleton_match"
    assert result["asr_spelling_variant_applied"] is True
    assert result["consonant_skeleton_similarity"] == 1.0
    assert result["vowel_tolerant_similarity"] >= 0.90


def test_asr_spelling_variant_updates_corrected_displayed_but_preserves_raw(tmp_path: Path) -> None:
    meta = {
        "expected_text": "hand",
        "raw_transcript": "hund",
        "corrected_transcript": "hund",
        "displayed_transcript": "hund",
        "accepted": False,
        "prompt_type": "word",
        "debug_metadata": {},
    }

    updated = apply_dynamic_expected_word_correction(meta, cmudict_loader=_loader(tmp_path))

    assert updated["raw_transcript"] == "hund"
    assert updated["corrected_transcript"] == "hand"
    assert updated["displayed_transcript"] == "hand"
    assert updated["accepted"] is True
    assert updated["correction_strategy_used"] == "dynamic_asr_spelling_variant"
    assert updated["asr_spelling_variant_applied"] is True


def test_asr_spelling_variant_rejects_unrelated_or_risky_words(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    banana = correct_expected_word("hand", "banana", "word", cmudict_loader=loader)
    head = correct_expected_word("hand", "head", "word", cmudict_loader=loader)
    function_word = correct_expected_word("the", "a", "word", cmudict_loader=loader)

    assert banana["accepted"] is False
    assert head["accepted"] is False
    assert function_word["accepted"] is False


def test_short_word_fragments_need_pronunciation_evidence(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    fish_weak = correct_expected_word("fish", "fs", "word", phoneme_similarity_score=0.40, cmudict_loader=loader)
    tree_weak = correct_expected_word("tree", "tr", "word", phoneme_similarity_score=0.50, cmudict_loader=loader)

    assert fish_weak["accepted"] is False
    assert fish_weak["suspicious_fragment"] is True
    assert fish_weak["sub_strategy"] == "rejected_suspicious_fragment_low_pronunciation_evidence"
    assert "suspicious_fragment" in fish_weak["reason"]
    assert tree_weak["accepted"] is False
    assert tree_weak["suspicious_fragment"] is True


def test_short_word_fragments_accept_only_with_strong_gop_or_phoneme_support(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    fish = correct_expected_word("fish", "fs", "word", gop_score=0.91, phoneme_similarity_score=0.83, cmudict_loader=loader)
    tree = correct_expected_word("tree", "tr", "word", phoneme_similarity_score=0.86, cmudict_loader=loader)

    assert fish["accepted"] is True
    assert fish["corrected_text"] == "fish"
    assert fish["display_text"] == "fish"
    assert fish["sub_strategy"] == "fragment_gop_supported_expected_match"
    assert tree["accepted"] is True
    assert tree["corrected_text"] == "tree"
    assert tree["sub_strategy"] == "fragment_phoneme_supported_expected_match"


def test_dynamic_safety_rejects_weak_or_short_function_word_matches(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    assert correct_expected_word("pulled", "poled", "passage", cmudict_loader=loader)["accepted"] is False
    assert correct_expected_word("the", "a", "passage", cmudict_loader=loader)["accepted"] is False


def test_sentence_word_alignment_marks_dynamic_and_homophone_acceptance(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    shield = dynamic_word_alignment("Arthur carried his shield", "arthur carried his shild", "passage", cmudict_loader=loader)
    knights = dynamic_word_alignment("Many knights pulled", "many nights pulled", "passage", cmudict_loader=loader)
    hand = dynamic_word_alignment("Raise your hand", "raise your hund", "passage", cmudict_loader=loader)

    assert shield[-1]["status"] == "accepted_by_asr_spelling_variant"
    assert shield[-1]["counts_as_correct"] is True
    assert knights[1]["status"] == "accepted_by_homophone"
    assert knights[1]["counts_as_correct"] is True
    assert hand[-1]["status"] == "accepted_by_asr_spelling_variant"
    assert hand[-1]["counts_as_correct"] is True


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
    assert updated["word_alignment"][-1]["status"] == "accepted_by_asr_spelling_variant"


def test_sentence_metadata_does_not_repair_phase_1b_split_merge_cases(tmp_path: Path) -> None:
    meta = {
        "expected_text": "time after lunch",
        "raw_transcript": "timeafter lunch",
        "corrected_transcript": "timeafter lunch",
        "displayed_transcript": "timeafter lunch",
        "prompt_type": "passage",
        "debug_metadata": {},
    }

    updated = apply_dynamic_expected_word_correction(meta, cmudict_loader=_loader(tmp_path))

    assert updated["raw_transcript"] == "timeafter lunch"
    assert updated["displayed_transcript"] == "timeafter lunch"
    assert not any(item.get("recognized_word") == "timeafter" and item.get("counts_as_correct") for item in updated["word_alignment"])


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
