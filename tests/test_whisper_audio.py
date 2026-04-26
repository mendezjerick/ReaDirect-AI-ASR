from pathlib import Path
from unittest.mock import patch

import pytest

from readirect_asr.finetuning.whisper_audio import load_audio_array


def test_load_audio_array_uses_librosa_backend(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    with patch("librosa.load", return_value=([0.0, 0.1], 16000)) as mocked_load:
        array, sr = load_audio_array(audio, sampling_rate=16000, backend="librosa")
    assert array == [0.0, 0.1]
    assert sr == 16000
    mocked_load.assert_called_once_with(str(audio), sr=16000, mono=True)


def test_load_audio_array_missing_file():
    with pytest.raises(FileNotFoundError):
        load_audio_array("missing.wav")


def test_load_audio_array_rejects_unknown_backend(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    with pytest.raises(ValueError):
        load_audio_array(audio, backend="torchcodec")
