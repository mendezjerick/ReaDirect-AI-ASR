from pathlib import Path

import numpy as np
import soundfile as sf

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


def _tone(path: Path, seconds: float = 1.2, sample_rate: int = 16000) -> Path:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    waveform = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(str(path), waveform.astype(np.float32), sample_rate)
    return path


class WrongLowConfidenceASR:
    provider = "wrong_low_confidence"
    model_size = "test-asr"

    def transcribe(self, audio_path: str, **kwargs):
        return {
            "transcript": "justice",
            "confidence": 0.1,
            "provider": self.provider,
            "model_used": self.model_size,
            "asr_route": "wav2vec2_only",
        }


def test_mock_asr_transcript_flows_into_reading_analyzer(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat"))
    assert response.ok is True
    assert response.transcript == "cat"
    assert response.error_type == "correct"
    assert response.trace == {}


def test_audio_trace_is_optional_and_expected_centric_when_requested(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(
        AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat", include_trace=True)
    )

    assert response.ok is True
    assert response.trace["final_transcript"] == "cat"
    assert response.trace["expected_centric"]["expected"] == "cat"
    assert response.trace["expected_centric"]["heard"] == "cat"
    assert response.trace["expected_centric"]["match"] is True
    assert response.trace["decoding"]["partial_steps"]
    assert response.trace_notes


def test_audio_analysis_scores_corrected_transcript_and_preserves_raw(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(
        AnalyzeAudioRequest(
            audio_path=str(audio),
            expected_text="Red",
            content_metadata={"mock_transcript": "Read"},
            debug=True,
        )
    )

    assert response.ok is True
    assert response.transcript == "Red"
    assert response.raw_transcript == "Read"
    assert response.corrected_transcript == "Red"
    assert response.displayed_transcript == "Red"
    assert response.normalized_transcript == "red"
    assert response.raw_wer == 1.0
    assert response.corrected_wer == 0.0
    assert response.is_correct is True
    assert response.normalization_applied is True
    assert response.accepted_by_phonetic_threshold is True
    assert response.debug_info["transcript_normalization"]["corrected_transcript"] == "Red"


def test_audio_analysis_applies_dynamic_expected_word_correction(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(
        AnalyzeAudioRequest(
            audio_path=str(audio),
            expected_text="shield",
            content_metadata={"mock_transcript": "shild"},
            debug=True,
        )
    )

    assert response.ok is True
    assert response.raw_transcript == "shild"
    assert response.corrected_transcript == "shield"
    assert response.displayed_transcript == "shield"
    assert response.dynamic_correction_applied is True
    assert response.correction_strategy_used == "dynamic_asr_spelling_variant"
    assert response.dynamic_correction_sub_strategy == "vowel_tolerant_consonant_skeleton_match"
    assert response.asr_spelling_variant_applied is True


def test_wrong_audible_low_confidence_transcript_is_not_retry_required(tmp_path: Path) -> None:
    audio = _tone(tmp_path / "valid-wrong.wav")
    base = _service(tmp_path)
    service = AIAnalysisService(WrongLowConfidenceASR(), base.cmudict_loader, base.content_repository, {
        "api": {"debug": True},
        "asr": {"provider": "wrong_low_confidence"},
        "audio_quality": {
            "min_duration_seconds": 1.0,
            "enable_quality_gate": True,
            "retry_on_bad_quality": True,
        },
        "transcript_normalization": {
            "low_confidence_threshold": 0.5,
        },
    })

    response = service.analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat", debug=True))

    assert response.ok is True
    assert response.raw_transcript == "justice"
    assert response.displayed_transcript == "justice"
    assert response.accepted is False
    assert response.is_correct is False
    assert response.retry_required is False
    assert response.quality_gate_failed is False
    assert "low_asr_confidence" not in response.uncertainty_reasons
    assert any("low" in note.lower() for note in response.developer_quality_notes)


def test_missing_audio_path_returns_safe_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(expected_text="cat"))
    assert response.ok is False
    assert response.error == "missing_audio_path"


def test_missing_audio_file_returns_safe_error(tmp_path: Path) -> None:
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(tmp_path / "missing.wav"), expected_text="cat"))
    assert response.ok is False
    assert response.error == "audio_file_not_found"


def test_unsupported_audio_type_returns_safe_error(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp4"
    audio.write_bytes(b"fake")
    response = _service(tmp_path).analyze_audio(AnalyzeAudioRequest(audio_path=str(audio), expected_text="cat", debug=True))
    assert response.ok is False
    assert response.error == "unsupported_audio_type"
    assert response.debug_info["supported_extension"] is False
