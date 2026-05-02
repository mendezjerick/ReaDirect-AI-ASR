from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
EPSILON = 1e-9

DEFAULT_AUDIO_QUALITY_CONFIG: dict[str, Any] = {
    "min_duration_seconds": 1.0,
    "max_duration_seconds": 30.0,
    "low_volume_dbfs": -35.0,
    "silence_dbfs": -40.0,
    "mostly_silent_ratio": 0.85,
    "clipping_threshold": 0.98,
    "clipped_ratio_threshold": 0.01,
    "min_speech_ratio": 0.15,
    "long_pause_seconds": 1.0,
    "very_long_pause_seconds": 2.0,
    "enable_quality_gate": True,
    "retry_on_bad_quality": True,
    "frame_length_ms": 25.0,
    "hop_length_ms": 10.0,
    "min_speech_segment_seconds": 0.05,
}


def audio_quality_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {**DEFAULT_AUDIO_QUALITY_CONFIG, **(config or {})}
    return {
        "min_duration_seconds": float(merged.get("min_duration_seconds", 1.0)),
        "max_duration_seconds": float(merged.get("max_duration_seconds", 30.0)),
        "low_volume_dbfs": float(merged.get("low_volume_dbfs", -35.0)),
        "silence_dbfs": float(merged.get("silence_dbfs", -40.0)),
        "mostly_silent_ratio": float(merged.get("mostly_silent_ratio", 0.85)),
        "clipping_threshold": float(merged.get("clipping_threshold", 0.98)),
        "clipped_ratio_threshold": float(merged.get("clipped_ratio_threshold", 0.01)),
        "min_speech_ratio": float(merged.get("min_speech_ratio", 0.15)),
        "long_pause_seconds": float(merged.get("long_pause_seconds", 1.0)),
        "very_long_pause_seconds": float(merged.get("very_long_pause_seconds", 2.0)),
        "enable_quality_gate": bool(merged.get("enable_quality_gate", True)),
        "retry_on_bad_quality": bool(merged.get("retry_on_bad_quality", True)),
        "frame_length_ms": float(merged.get("frame_length_ms", 25.0)),
        "hop_length_ms": float(merged.get("hop_length_ms", 10.0)),
        "min_speech_segment_seconds": float(merged.get("min_speech_segment_seconds", 0.05)),
    }


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


def load_audio_for_quality(audio_path: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waveform, loaded_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    return waveform, int(loaded_sample_rate)


def rms_to_dbfs(rms: float) -> float:
    return float(20.0 * np.log10(float(rms) + EPSILON))


def validate_minimum_duration(duration_seconds: float, config: dict[str, Any] | None = None) -> bool:
    cfg = audio_quality_config(config)
    return float(duration_seconds) >= cfg["min_duration_seconds"]


def detect_low_volume(waveform: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = audio_quality_config(config)
    if waveform.size == 0:
        rms = 0.0
    else:
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    rms_dbfs = rms_to_dbfs(rms)
    return {
        "rms": round(rms, 6),
        "rms_dbfs": round(rms_dbfs, 3),
        "low_volume": rms_dbfs < cfg["low_volume_dbfs"],
    }


def detect_clipping(waveform: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = audio_quality_config(config)
    if waveform.size == 0:
        ratio = 0.0
        peak = 0.0
    else:
        absolute = np.abs(waveform)
        ratio = float(np.count_nonzero(absolute >= cfg["clipping_threshold"]) / waveform.size)
        peak = float(np.max(absolute))
    return {
        "peak_amplitude": round(peak, 6),
        "clipped_sample_ratio": round(ratio, 6),
        "clipped": ratio >= cfg["clipped_ratio_threshold"],
    }


def detect_speech_segments(
    waveform: np.ndarray,
    sample_rate: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    cfg = audio_quality_config(config)
    if waveform.size == 0 or sample_rate <= 0:
        return []

    frame_length = max(1, int(sample_rate * cfg["frame_length_ms"] / 1000.0))
    hop_length = max(1, int(sample_rate * cfg["hop_length_ms"] / 1000.0))
    threshold = float(10.0 ** (cfg["silence_dbfs"] / 20.0))
    min_segment = cfg["min_speech_segment_seconds"]

    frame_rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    voiced = frame_rms >= threshold
    if not np.any(voiced):
        return []

    segments: list[dict[str, float]] = []
    start_frame: int | None = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and start_frame is None:
            start_frame = index
        if start_frame is not None and (not is_voiced or index == len(voiced) - 1):
            end_frame = index if not is_voiced else index + 1
            start_seconds = start_frame * hop_length / sample_rate
            end_seconds = min((end_frame * hop_length + frame_length) / sample_rate, waveform.size / sample_rate)
            duration_seconds = max(0.0, end_seconds - start_seconds)
            if duration_seconds >= min_segment:
                segments.append(
                    {
                        "start_seconds": round(start_seconds, 3),
                        "end_seconds": round(end_seconds, 3),
                        "duration_seconds": round(duration_seconds, 3),
                    }
                )
            start_frame = None
    return segments


def detect_pause_statistics(
    speech_segments: list[dict[str, float]],
    duration_seconds: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = audio_quality_config(config)
    pauses: list[float] = []
    sorted_segments = sorted(speech_segments, key=lambda item: float(item.get("start_seconds", 0.0)))
    for previous, current in zip(sorted_segments, sorted_segments[1:]):
        gap = float(current.get("start_seconds", 0.0)) - float(previous.get("end_seconds", 0.0))
        if gap > 0:
            pauses.append(gap)

    total_pause = float(sum(pauses))
    longest_pause = float(max(pauses) if pauses else 0.0)
    duration = max(float(duration_seconds), EPSILON)
    return {
        "speech_segment_count": len(sorted_segments),
        "pause_count": len(pauses),
        "long_pause_count": sum(1 for pause in pauses if pause >= cfg["long_pause_seconds"]),
        "very_long_pause_count": sum(1 for pause in pauses if pause >= cfg["very_long_pause_seconds"]),
        "longest_pause_seconds": round(longest_pause, 3),
        "total_pause_seconds": round(total_pause, 3),
        "pause_ratio": round(total_pause / duration, 3),
    }


def detect_silence_ratio(
    waveform: np.ndarray,
    sample_rate: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration_seconds = float(waveform.size / sample_rate) if sample_rate > 0 else 0.0
    speech_segments = detect_speech_segments(waveform, sample_rate, config)
    speech_duration = float(sum(segment["duration_seconds"] for segment in speech_segments))
    if duration_seconds <= 0:
        speech_ratio = 0.0
    else:
        speech_ratio = min(1.0, speech_duration / duration_seconds)
    silence_ratio = max(0.0, 1.0 - speech_ratio)
    return {
        "silence_ratio": round(silence_ratio, 3),
        "speech_ratio": round(speech_ratio, 3),
        "speech_duration_seconds": round(speech_duration, 3),
        "speech_segments": speech_segments,
    }


def analyze_audio_quality(audio_path: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = audio_quality_config(config)
    path = Path(audio_path)
    base_report = validate_audio_file(path)
    warnings_list: list[str] = []
    quality_flags = {
        "too_short": False,
        "mostly_silent": False,
        "low_volume": False,
        "clipped": False,
        "no_speech_detected": False,
    }

    if not base_report["exists"] or not base_report["supported_extension"]:
        warnings_list.extend(str(warning) for warning in base_report.get("warnings", []))
        return {
            "audio_valid": False,
            "duration_seconds": base_report.get("duration_seconds"),
            "sample_rate": None,
            "rms": 0.0,
            "rms_dbfs": 0.0,
            "peak_amplitude": 0.0,
            "clipped_sample_ratio": 0.0,
            "silence_ratio": 1.0,
            "speech_ratio": 0.0,
            "speech_duration_seconds": 0.0,
            "speech_segment_count": 0,
            "long_pause_count": 0,
            "longest_pause_seconds": 0.0,
            "pause_ratio": 0.0,
            "speech_segments": [],
            "pause_metrics": detect_pause_statistics([], 0.0, cfg),
            "warnings": warnings_list,
            "quality_flags": quality_flags,
        }

    try:
        waveform, sample_rate = load_audio_for_quality(path)
    except Exception:
        duration = base_report.get("duration_seconds")
        warnings_list.extend(str(warning) for warning in base_report.get("warnings", []))
        warnings_list.append("audio_quality_unreadable")
        return {
            "audio_valid": False,
            "duration_seconds": duration,
            "sample_rate": None,
            "rms": 0.0,
            "rms_dbfs": 0.0,
            "peak_amplitude": 0.0,
            "clipped_sample_ratio": 0.0,
            "silence_ratio": 0.0,
            "speech_ratio": 0.0,
            "speech_duration_seconds": 0.0,
            "speech_segment_count": 0,
            "long_pause_count": 0,
            "longest_pause_seconds": 0.0,
            "pause_ratio": 0.0,
            "speech_segments": [],
            "pause_metrics": detect_pause_statistics([], float(duration or 0.0), cfg),
            "warnings": warnings_list,
            "quality_flags": quality_flags,
        }

    duration_seconds = round(float(waveform.size / sample_rate) if sample_rate > 0 else 0.0, 3)
    volume = detect_low_volume(waveform, cfg)
    clipping = detect_clipping(waveform, cfg)
    speech = detect_silence_ratio(waveform, sample_rate, cfg)
    speech_segments = list(speech["speech_segments"])
    pause_metrics = detect_pause_statistics(speech_segments, duration_seconds, cfg)

    quality_flags["too_short"] = not validate_minimum_duration(duration_seconds, cfg)
    quality_flags["low_volume"] = bool(volume["low_volume"])
    quality_flags["clipped"] = bool(clipping["clipped"])
    quality_flags["no_speech_detected"] = len(speech_segments) == 0
    quality_flags["mostly_silent"] = (
        float(speech["silence_ratio"]) >= cfg["mostly_silent_ratio"]
        or float(speech["speech_ratio"]) < cfg["min_speech_ratio"]
    )

    if quality_flags["too_short"]:
        warnings_list.append("audio_too_short")
    if duration_seconds > cfg["max_duration_seconds"]:
        warnings_list.append("audio_too_long")
    if quality_flags["no_speech_detected"]:
        warnings_list.append("no_speech_detected")
    if quality_flags["mostly_silent"]:
        warnings_list.append("mostly_silent")
    if quality_flags["low_volume"]:
        warnings_list.append("low_volume")
    if quality_flags["clipped"]:
        warnings_list.append("clipped")

    audio_valid = not any(quality_flags.values())
    return {
        "audio_valid": audio_valid,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "rms": volume["rms"],
        "rms_dbfs": volume["rms_dbfs"],
        "peak_amplitude": clipping["peak_amplitude"],
        "clipped_sample_ratio": clipping["clipped_sample_ratio"],
        "silence_ratio": speech["silence_ratio"],
        "speech_ratio": speech["speech_ratio"],
        "speech_duration_seconds": speech["speech_duration_seconds"],
        "speech_segment_count": len(speech_segments),
        "long_pause_count": pause_metrics["long_pause_count"],
        "longest_pause_seconds": pause_metrics["longest_pause_seconds"],
        "pause_ratio": pause_metrics["pause_ratio"],
        "speech_segments": speech_segments,
        "pause_metrics": pause_metrics,
        "warnings": warnings_list,
        "quality_flags": quality_flags,
    }


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
