from pathlib import Path

from readirect_asr.text.reinforcement_corrections import load_reinforcement_corrections
from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


def _normalize(expected: str, raw: str, config: dict | None = None):
    return normalize_asr_transcript(raw_transcript=raw, expected_text=expected, config=config or {})


def test_load_letter_reinforcement_csv_case_insensitive() -> None:
    table = load_reinforcement_corrections()

    assert "letter-reinforcement.csv" in table.files_loaded
    assert table.letter_rules_count == 20
    assert table.word_rules_count == 0
    assert table.match("z", "they", "letter") is not None
    assert table.match(" Z ", " They ", "letter") is not None
    assert table.match("W", "zavil you", "letter") is not None


def test_expected_z_raw_they_accepted_by_reinforcement() -> None:
    result = _normalize("Z", "They")

    assert result.accepted is True
    assert result.corrected_transcript == "Z"
    assert result.displayed_transcript == "Z"
    assert result.accepted_by_reinforcement_match is True
    assert result.correction_strategy_used == "reinforcement_error_transcript_match"
    assert result.reinforcement_expected_label == "Z"
    assert result.reinforcement_matched_transcript == "They"


def test_letter_reinforcement_accepts_curated_rows() -> None:
    for expected, raw in [
        ("O", "Oh"),
        ("C", "See"),
        ("Q", "Cue"),
        ("W", "Zavil you"),
        ("W", "Zebel you"),
    ]:
        result = _normalize(expected, raw)

        assert result.accepted is True, (expected, raw, result.normalization_reason)
        assert result.corrected_transcript == expected
        assert result.displayed_transcript == expected
        assert result.accepted_by_reinforcement_match is True


def test_reinforcement_rejects_unmatched_and_non_expected_centric_cases() -> None:
    banana = _normalize("Z", "Banana")
    wrong_expected = _normalize("C", "They")

    assert banana.accepted is False
    assert banana.corrected_transcript == "Banana"
    assert banana.displayed_transcript == "Banana"
    assert banana.accepted_by_reinforcement_match is False

    assert wrong_expected.accepted is False
    assert wrong_expected.corrected_transcript == "They"
    assert wrong_expected.displayed_transcript == "They"
    assert wrong_expected.accepted_by_reinforcement_match is False


def test_reinforcement_disabled_ignores_table_and_continues_normal_pipeline() -> None:
    result = _normalize("Z", "They", {"reinforcement_corrections_enabled": False})

    assert result.accepted_by_reinforcement_match is False
    assert result.correction_strategy_used != "reinforcement_error_transcript_match"
    assert result.debug_metadata["reinforcement_corrections"]["reinforcement_corrections_enabled"] is False


def test_missing_reinforcement_file_does_not_crash(tmp_path: Path) -> None:
    result = _normalize(
        "O",
        "Oh",
        {
            "reinforcement_corrections_enabled": True,
            "reinforcement_corrections_dir": str(tmp_path),
            "letter_reinforcement_file": "missing.csv",
        },
    )

    assert result.accepted_by_reinforcement_match is False
    assert result.accepted is True
    warnings = result.debug_metadata["reinforcement_corrections"]["reinforcement_load_warnings"]
    assert any("Letter reinforcement file not found" in warning for warning in warnings)
