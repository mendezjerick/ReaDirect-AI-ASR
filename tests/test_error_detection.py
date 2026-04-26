from readirect_asr.scoring.error_detection import detect_error_type


def test_detect_correct() -> None:
    assert detect_error_type("cat", "cat")["error_type"] == "correct"


def test_detect_blank() -> None:
    assert detect_error_type("cat", "")["error_type"] == "blank"


def test_detect_initial_sound_error() -> None:
    result = detect_error_type("cat", "bat", expected_phonemes=["K", "AE", "T"], actual_phonemes=["B", "AE", "T"])
    assert result["error_type"] == "initial_sound_error"


def test_detect_final_sound_error() -> None:
    result = detect_error_type("cat", "cap", expected_phonemes=["K", "AE", "T"], actual_phonemes=["K", "AE", "P"])
    assert result["error_type"] == "final_sound_error"


def test_detect_vowel_error() -> None:
    result = detect_error_type("cat", "cut", expected_phonemes=["K", "AE", "T"], actual_phonemes=["K", "AH", "T"])
    assert result["error_type"] == "vowel_error"


def test_detect_sentence_errors() -> None:
    assert detect_error_type("red cat sat", "red cat")["error_type"] in {"skipped_word", "partial_sentence"}
    assert detect_error_type("red cat sat", "cat red sat")["error_type"] == "word_order_error"
    assert detect_error_type("red cat sat", "zebra")["error_type"] == "partial_sentence"

