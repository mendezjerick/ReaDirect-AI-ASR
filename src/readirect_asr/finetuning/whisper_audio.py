from __future__ import annotations

from pathlib import Path
from typing import Any


def load_audio_array(audio_path: str | Path, sampling_rate: int = 16000, backend: str = "librosa") -> tuple[Any, int]:
    """Load audio for Whisper without datasets.Audio/TorchCodec.

    The default backend is librosa because it avoids the TorchCodec/FFmpeg DLL path
    that is brittle on Windows.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if backend != "librosa":
        raise ValueError(f"Unsupported audio_loading_backend: {backend}")
    import librosa

    audio, sr = librosa.load(str(path), sr=sampling_rate, mono=True)
    return audio, int(sr)
