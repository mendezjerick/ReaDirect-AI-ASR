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
                "SUN S AH1 N",
                "SON S AH1 N",
                "TIME T AY1 M",
                "AFTER AE1 F T ER0",
                "OPEN OW1 P AH0 N",
                "MAYA M AY1 AH0",
                "SMILED S M AY1 L D",
                "REEDS R IY1 D Z",
                "SHE SH IY1",
                "FRUIT F R UW1 T",
                "SEEDS S IY1 D Z",
                "NEARBY N IH1 R B AY2",
                "NEAR N IH1 R",
                "BY B AY1",
                "BOOK B UH1 K",
                "MARK M AA1 R K",
                "WOVEN W OW1 V AH0 N",
                "WOMAN W UH1 M AH0 N",
            ]
        ),
        encoding="utf-8",
    )
    (cmu / "cmudict.phones").write_text("N nasal\nAY vowel\nT stop\nS fricative\nP stop\nUH vowel\nL liquid\nD stop\nOW vowel\nDH fricative\nAH vowel\nF fricative\nIH vowel\nSH fricative\nR liquid\nIY vowel\nHH aspirate\nAE vowel\nEH vowel\nER vowel\nM nasal\nB stop\nUW vowel\nK stop\nAA vowel\nW semivowel\nV fricative\nZ fricative\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("N\nAY\nAY1\nAY2\nT\nS\nP\nUH\nUH1\nL\nD\nOW\nOW1\nDH\nAH\nAH0\nAH1\nF\nIH\nIH1\nSH\nR\nIY\nIY1\nHH\nAE\nAE1\nEH\nEH1\nER\nER0\nM\nB\nUW\nUW1\nK\nAA\nAA1\nW\nV\nZ\n", encoding="utf-8")
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
    common_homophone = correct_expected_word("sun", "son", "word", cmudict_loader=loader)
    common_homophone_without_phoneme = correct_expected_word("sun", "son", "word", phoneme_similarity_score=0.0, cmudict_loader=loader)

    assert spelling["accepted"] is True
    assert spelling["sub_strategy"] == "vowel_tolerant_consonant_skeleton_match"
    assert homophone["accepted"] is True
    assert homophone["homophone_match"] is True
    assert common_homophone["accepted"] is True
    assert common_homophone["corrected_text"] == "sun"
    assert common_homophone["sub_strategy"] in {"homophone_match", "known_asr_confusion_match"}
    assert common_homophone_without_phoneme["accepted"] is True
    assert common_homophone_without_phoneme["sub_strategy"] == "known_asr_confusion_match"


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


def test_passage_alignment_repairs_two_expected_words_to_one_raw_chunk(tmp_path: Path) -> None:
    alignment = dynamic_word_alignment("time after lunch", "timeafter lunch", "passage", cmudict_loader=_loader(tmp_path))
    repaired = [item for item in alignment if item["expected_word"] in {"time", "after"}]

    assert len(repaired) == 2
    assert {item["status"] for item in repaired} == {"accepted_by_split_merge"}
    assert all(item["recognized_word"] == "timeafter" for item in repaired)
    assert all(item["operation"] == "merge_match" for item in repaired)
    assert all(item["counts_as_correct"] is True for item in repaired)


def test_passage_alignment_repairs_common_merge_and_split_cases(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    cases = [
        ("open Maya", "openmya", {"open", "maya"}),
        ("smiled the", "smiledthe", {"smiled", "the"}),
        ("reeds she", "reedsshe", {"reeds", "she"}),
        ("book mark", "bookmark", {"book", "mark"}),
    ]

    for expected, raw, expected_words in cases:
        alignment = dynamic_word_alignment(expected, raw, "passage", cmudict_loader=loader)
        repaired = [item for item in alignment if item["expected_word"] in expected_words]

        assert repaired
        assert all(item["status"] == "accepted_by_split_merge" for item in repaired)
        assert all(item["counts_as_correct"] is True for item in repaired)

    split = dynamic_word_alignment("nearby", "near by", "passage", cmudict_loader=loader)

    assert split[0]["status"] == "accepted_by_split_merge"
    assert split[0]["recognized_word"] == "near by"
    assert split[0]["operation"] == "split_match"
    assert split[0]["counts_as_correct"] is True


def test_passage_alignment_handles_distorted_boundary_repair_and_risky_rejections(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    fruit = dynamic_word_alignment("fruit seeds", "fruitsieds", "passage", cmudict_loader=loader)
    woven = dynamic_word_alignment("woven", "woman", "passage", cmudict_loader=loader)
    function_word = dynamic_word_alignment("the", "a", "passage", cmudict_loader=loader)

    assert all(item["status"] in {"accepted_by_split_merge", "partial"} for item in fruit)
    assert any(item["status"] == "accepted_by_split_merge" for item in fruit)
    assert woven[0]["status"] in {"incorrect", "partial"}
    assert woven[0]["counts_as_correct"] is False
    assert function_word[0]["status"] == "incorrect"
    assert function_word[0]["counts_as_correct"] is False


def test_sentence_metadata_repairs_phase_1b_split_merge_cases(tmp_path: Path) -> None:
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
    assert any(item.get("recognized_word") == "timeafter" and item.get("counts_as_correct") for item in updated["word_alignment"])
    assert updated["word_alignment"][0]["status"] == "accepted_by_split_merge"
    assert "alignment_debug" in updated["debug_metadata"]


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
