from readirect_asr.content.difficulty import compute_difficulty


def test_easy_word_gets_easy_or_very_easy() -> None:
    result = compute_difficulty({"expected_text": "cat"}, {"item_text_type": "single_word", "phoneme_count": 3, "syllable_estimate": 1, "phoneme_pattern": "CVC"})
    assert result["difficulty_level"] in {"very_easy", "easy"}


def test_longer_sentence_gets_higher_difficulty() -> None:
    short = compute_difficulty({"expected_text": "cat sat"}, {"item_text_type": "sentence"})
    long = compute_difficulty({"expected_text": "The little animal walked across the classroom and looked at every window."}, {"item_text_type": "sentence"})
    assert long["difficulty_score"] > short["difficulty_score"]


def test_missing_cmudict_increases_review_risk_factor() -> None:
    result = compute_difficulty({"expected_text": "zzzz"}, {"item_text_type": "single_word", "phoneme_count": 0, "syllable_estimate": 1, "cmudict_missing_words": "zzzz"})
    assert result["difficulty_factors"]["missing_cmudict"] > 0

