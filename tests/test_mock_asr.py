from readirect_asr.asr.mock_asr import MockASR


def test_mock_asr_returns_expected_text_when_provided() -> None:
    result = MockASR().transcribe("fake.wav", expected_text="cat")
    assert result["transcript"] == "cat"
    assert result["confidence"] is None
    assert result["provider"] == "mock"


def test_mock_asr_returns_placeholder_without_expected_text() -> None:
    result = MockASR().transcribe("fake.wav")
    assert result["transcript"] == "mock transcript"

