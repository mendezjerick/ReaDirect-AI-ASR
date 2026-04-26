from readirect_asr.scoring.text_similarity import (
    classify_similarity,
    levenshtein_distance,
    normalize_text,
    similarity_percentage,
)


def test_normalize_text_lowercases_and_removes_punctuation() -> None:
    assert normalize_text("  Cat, CAT!  ") == "cat cat"


def test_levenshtein_distance_uses_normalized_text() -> None:
    assert levenshtein_distance("cat", "cut") == 1


def test_similarity_percentage_exact_match() -> None:
    assert similarity_percentage("cat", "cat") == 100.0


def test_classify_similarity_labels_blank_and_exact() -> None:
    assert classify_similarity("cat", "") == "blank"
    assert classify_similarity("cat", "CAT!") == "exact"


def test_classify_similarity_far() -> None:
    assert classify_similarity("cat", "zebra") == "far"

