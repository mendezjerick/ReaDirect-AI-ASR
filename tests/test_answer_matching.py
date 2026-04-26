from readirect_asr.scoring.answer_matching import match_answer, parse_accepted_answers


def test_exact_match_is_correct() -> None:
    result = match_answer("Cat!", "cat")
    assert result["is_exact"] is True
    assert result["is_correct"] is True
    assert result["similarity_label"] == "exact"


def test_accepted_answer_is_correct() -> None:
    result = match_answer("cat", "kitty", ["kitty"])
    assert result["is_accepted"] is True
    assert result["is_correct"] is True


def test_close_answer_not_automatically_correct() -> None:
    result = match_answer("cat", "cap")
    assert result["is_correct"] is False
    assert result["similarity_label"] == "very_close"


def test_blank_answer_returns_blank() -> None:
    result = match_answer("cat", "")
    assert result["similarity_label"] == "blank"


def test_parse_accepted_answers() -> None:
    assert parse_accepted_answers("cat|kitty") == ["cat", "kitty"]
    assert parse_accepted_answers('["cat", "kitty"]') == ["cat", "kitty"]

