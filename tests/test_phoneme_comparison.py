from readirect_asr.scoring.phoneme_comparison import compare_phonemes, phoneme_similarity


def test_cat_vs_cap_detects_final_mismatch() -> None:
    result = compare_phonemes(["K", "AE", "T"], ["K", "AE", "P"])
    assert result["initial_phoneme_match"] is True
    assert result["final_phoneme_match"] is False
    assert result["vowel_phoneme_match"] is True


def test_cat_vs_bat_detects_initial_mismatch() -> None:
    result = compare_phonemes(["K", "AE", "T"], ["B", "AE", "T"])
    assert result["initial_phoneme_match"] is False


def test_cat_vs_cut_detects_vowel_mismatch() -> None:
    result = compare_phonemes(["K", "AE", "T"], ["K", "AH", "T"])
    assert result["vowel_phoneme_match"] is False


def test_phoneme_similarity_and_missing() -> None:
    assert phoneme_similarity(["K", "AE", "T"], ["K", "AE", "P"]) == 0.666667
    assert phoneme_similarity([], ["K"]) == 0.0

