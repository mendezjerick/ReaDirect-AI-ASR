import json
from pathlib import Path

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.reading_analyzer import analyze_reading_response


def _loader(tmp_path: Path) -> CMUDictLoader:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text(
        "CAT K AE1 T\nCAP K AE1 P\nBAT B AE1 T\nCUT K AH1 T\nRED R EH1 D\nSAT S AE1 T\n",
        encoding="utf-8",
    )
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\nP stop\nB stop\nAH vowel\nR liquid\nEH vowel\nD stop\nS fricative\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\nP\nB\nAH\nAH1\nR\nEH\nEH1\nD\nS\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def test_cat_cap_returns_final_sound_error(tmp_path: Path) -> None:
    result = analyze_reading_response("cat", "cap", cmudict_loader=_loader(tmp_path))
    assert result["error_type"] == "final_sound_error"
    assert result["skill_signal"] == "final_consonant"
    assert result["feedback_hint"] == "ending_sound"


def test_cat_cat_returns_correct(tmp_path: Path) -> None:
    result = analyze_reading_response("cat", "cat", cmudict_loader=_loader(tmp_path))
    assert result["is_correct"] is True
    assert result["error_type"] == "correct"


def test_sentence_missing_word_and_json_serializable(tmp_path: Path) -> None:
    result = analyze_reading_response("red cat sat", "red cat", cmudict_loader=_loader(tmp_path))
    assert result["error_type"] in {"skipped_word", "partial_sentence"}
    json.dumps(result)

