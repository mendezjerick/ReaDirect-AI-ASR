from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from api.schemas import AnalyzeAudioRequest
from api.service import AIAnalysisService
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.audio.preprocessing import analyze_audio_quality
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


SAMPLE_RATE = 16000


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    sf.write(str(path), waveform.astype(np.float32), sample_rate)
    return path


def _tone(seconds: float, amplitude: float = 0.2, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * 220.0 * t)


def _service(tmp_path: Path) -> AIAnalysisService:
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("CAT K AE1 T\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("K stop\nAE vowel\nT stop\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("K\nAE\nAE1\nT\n", encoding="utf-8")
    loader = CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()
    repo = ContentRepository(tmp_path / "missing.csv", tmp_path / "missing2.csv").load()
    return AIAnalysisService(
        MockASR(),
        loader,
        repo,
        {
            "api": {"debug": True},
            "asr": {"provider": "mock"},
            "audio_quality": {
                "min_duration_seconds": 1.0,
                "enable_quality_gate": True,
                "retry_on_bad_quality": True,
            },
        },
    )


def test_valid_audio_quality_does_not_require_retry(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "valid.wav", _tone(1.2))
    report = analyze_audio_quality(audio)
    assert report["audio_valid"] is True
    assert report["quality_flags"]["too_short"] is False
    assert report["quality_flags"]["no_speech_detected"] is False

    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat"))
    assert response.audio_quality["audio_valid"] is True
    assert response.retry_required is False
    assert response.quality_gate_failed is False


def test_too_short_audio_requires_retry(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "short.wav", _tone(0.25))
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat"))
    assert response.audio_quality["quality_flags"]["too_short"] is True
    assert response.retry_required is True
    assert response.quality_gate_failed is True
    assert response.accepted is False
    assert response.displayed_transcript == ""


def test_silent_audio_requires_retry(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "silent.wav", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat"))
    flags = response.audio_quality["quality_flags"]
    assert flags["mostly_silent"] is True or flags["no_speech_detected"] is True
    assert response.retry_required is True


def test_low_volume_audio_sets_warning(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "quiet.wav", _tone(1.2, amplitude=0.003))
    report = analyze_audio_quality(audio, {"silence_dbfs": -60.0})
    assert report["quality_flags"]["low_volume"] is True
    assert "low_volume" in report["warnings"]


def test_clipped_audio_sets_warning(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "clipped.wav", np.ones(SAMPLE_RATE * 2, dtype=np.float32))
    report = analyze_audio_quality(audio)
    assert report["quality_flags"]["clipped"] is True
    assert "clipped" in report["warnings"]


def test_pause_detection_reports_gap(tmp_path: Path) -> None:
    waveform = np.concatenate(
        [
            _tone(0.6),
            np.zeros(int(SAMPLE_RATE * 1.2), dtype=np.float32),
            _tone(0.6),
        ]
    )
    audio = _write_wav(tmp_path / "pause.wav", waveform)
    report = analyze_audio_quality(audio)
    pause_metrics = report["pause_metrics"]
    assert pause_metrics["speech_segment_count"] >= 2
    assert pause_metrics["pause_count"] >= 1
    assert pause_metrics["long_pause_count"] >= 1


def test_asr_response_schema_includes_quality_fields(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "valid.wav", _tone(1.2))
    data = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat")).model_dump()
    for field in [
        "audio_quality",
        "pause_metrics",
        "uncertain",
        "retry_required",
        "uncertainty_reasons",
        "quality_gate_failed",
        "learner_retry_message",
    ]:
        assert field in data
