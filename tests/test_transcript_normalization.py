from readirect_asr.text.normalization import (
    normalize_for_wer,
    normalize_transcript,
    normalize_whitespace,
    remove_common_hesitations,
    remove_punctuation,
)


def test_remove_punctuation() -> None:
    assert remove_punctuation("Cat, dog!") == "Cat  dog "


def test_normalize_whitespace() -> None:
    assert normalize_whitespace(" a   b \n c ") == "a b c"


def test_remove_common_hesitations() -> None:
    assert remove_common_hesitations("um cat uh dog") == "cat dog"


def test_normalize_transcript_lowercases_and_cleans() -> None:
    assert normalize_transcript(" Cat, DOG! ") == "cat dog"


def test_normalize_for_wer_removes_hesitations() -> None:
    assert normalize_for_wer("Um, cat!") == "cat"

