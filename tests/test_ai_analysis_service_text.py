from pathlib import Path

import pandas as pd

from api.schemas import AnalyzeTextRequest
from api.service import AIAnalysisService
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _loader(tmp_path: Path) -> CMUDictLoader:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("CAT K AE1 T\nCAP K AE1 P\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\nP stop\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\nP\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def _service(tmp_path: Path) -> AIAnalysisService:
    index = tmp_path / "content_index.csv"
    pd.DataFrame([{"prompt_id": "M2-001", "expected_text": "cat", "accepted_answers": "cat"}]).to_csv(index, index=False)
    repo = ContentRepository(index, tmp_path / "missing.csv").load()
    return AIAnalysisService(MockASR(), _loader(tmp_path), repo, {"api": {"debug": True}, "asr": {"provider": "mock"}})


def test_analyze_text_cat_cap_returns_final_sound_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_text(AnalyzeTextRequest(expected_text="cat", actual_text="cap"))
    assert response.ok is True
    assert response.error_type == "final_sound_error"


def test_accepted_answer_returns_correct(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_text(AnalyzeTextRequest(expected_text="cat", actual_text="kitty", accepted_answers=["kitty"]))
    assert response.is_correct is True
    assert response.error_type == "accepted_variant"


def test_missing_expected_text_returns_safe_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_text(AnalyzeTextRequest(actual_text="cat"))
    assert response.ok is False
    assert response.error == "missing_expected_text"


def test_prompt_id_lookup_fills_expected_text(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_text(AnalyzeTextRequest(prompt_id="M2-001", actual_text="cat"))
    assert response.expected_text == "cat"
    assert response.is_correct is True

