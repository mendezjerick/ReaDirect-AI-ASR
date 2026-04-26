from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import soundfile as sf

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}


def is_supported_audio_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def get_audio_duration_seconds(audio_path: str | Path) -> float | None:
    path = Path(audio_path)
    if not path.exists() or not is_supported_audio_file(path):
        return None
    try:
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return round(float(info.frames) / float(info.samplerate), 3)
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return round(float(librosa.get_duration(path=str(path))), 3)
    except Exception:
        return None


def validate_audio_file(audio_path: str | Path) -> dict[str, object]:
    path = Path(audio_path)
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "supported_extension": is_supported_audio_file(path),
        "duration_seconds": None,
        "warnings": [],
    }
    warnings = result["warnings"]
    assert isinstance(warnings, list)

    if not result["exists"]:
        warnings.append("audio file is missing")
        return result
    if not result["supported_extension"]:
        warnings.append(f"unsupported audio extension: {path.suffix.lower()}")
        return result

    duration = get_audio_duration_seconds(path)
    result["duration_seconds"] = duration
    if duration is None:
        warnings.append("audio duration could not be read")
    return result


def list_audio_files(audio_dir: str | Path) -> list[Path]:
    base = Path(audio_dir)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file() and is_supported_audio_file(path))


def describe_preprocessing_plan(audio_dir: str | Path, output_dir: str | Path) -> list[str]:
    files = list_audio_files(audio_dir)
    return [
        f"Would prepare {path} -> {Path(output_dir) / path.name}"
        for path in files
    ]
