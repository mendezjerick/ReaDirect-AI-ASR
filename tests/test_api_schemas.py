from api.schemas import AnalysisResponse, AnalyzeAudioRequest, AnalyzeTextRequest


def test_analyze_text_request_validates() -> None:
    request = AnalyzeTextRequest(expected_text="cat", actual_text="cap")
    assert request.expected_text == "cat"


def test_analyze_audio_request_validates() -> None:
    request = AnalyzeAudioRequest(audio_path="sample.wav", expected_text="cat")
    assert request.audio_path == "sample.wav"


def test_analysis_response_serializes() -> None:
    response = AnalysisResponse(ok=True, request_id="r1", mode="text", provider="mock")
    data = response.model_dump()
    assert data["ok"] is True
    assert data["warnings"] == []

