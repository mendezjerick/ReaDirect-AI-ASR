from pathlib import Path

from api.schemas import AnalyzeAudioRequest
from api.service import AIAnalysisService
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _service(tmp_path: Path) -> AIAnalysisService:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("CAT K AE1 T\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\n", encoding="utf-8")
    loader = CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()
    repo = ContentRepository(tmp_path / "missing.csv", tmp_path / "missing2.csv").load()
    return AIAnalysisService(MockASR(), loader, repo, {"api": {"debug": True}, "asr": {"provider": "mock"}})


def test_mock_asr_transcript_flows_into_reading_analyzer(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat"))
    assert response.ok is True
    assert response.transcript == "cat"
    assert response.error_type == "correct"


def test_missing_audio_path_returns_safe_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(expected_text="cat"))
    assert response.ok is False
    assert response.error == "missing_audio_path"


def test_missing_audio_file_returns_safe_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(tmp_path / "missing.wav"), expected_text="cat"))
    assert response.ok is False
    assert response.error == "audio_file_not_found"

