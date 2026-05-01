from readirect_asr.audio.preprocessing import is_supported_audio_file, validate_audio_file


def test_supported_extension_detection() -> None:
    assert is_supported_audio_file("sample.wav")
    assert is_supported_audio_file("sample.flac")
    assert is_supported_audio_file("sample.webm")
    assert not is_supported_audio_file("sample.mp4")
    assert not is_supported_audio_file("sample.weba")
    assert not is_supported_audio_file("sample.txt")


def test_missing_audio_returns_warning() -> None:
    report = validate_audio_file("missing.wav")
    assert report["exists"] is False
    assert "audio file is missing" in report["warnings"]
