from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from training.text_normalization import normalize_asr_text
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        result = os.getenv(name, default)
        if result is None:
            raise RuntimeError(f"Required environment variable is not set: {name}")
        return result

    return ENV_PATTERN.sub(replace, value)


def load_alpha_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Alpha config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _expand_env_value(config)


def _discover_windows_ffmpeg_bin() -> Path | None:
    candidates: list[Path] = []
    executable = shutil.which("ffmpeg")
    if executable:
        candidates.append(Path(executable).resolve().parent)

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        candidates.extend(
            path
            for pattern in (
                "Gyan.FFmpeg.Shared_*/*/bin",
                "BtbN.FFmpeg.*.Shared_*/*/bin",
            )
            for path in winget_packages.glob(pattern)
        )

    for directory in candidates:
        if (directory / "ffmpeg.exe").exists() and any(directory.glob("avcodec-*.dll")):
            return directory.resolve()
    return None


def configure_windows_ffmpeg() -> Path | None:
    value = os.getenv("FFMPEG_BIN_DIR", "").strip().strip("\"'")
    directory = Path(value).expanduser() if value else None
    if directory is not None and directory.is_file() and directory.name.lower() == "ffmpeg.exe":
        directory = directory.parent
    if directory is not None and not directory.is_dir():
        directory = None
    if directory is None:
        directory = _discover_windows_ffmpeg_bin()
    if directory is None:
        return None
    directory = directory.resolve()
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(directory))
    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
    return directory


def find_gigaspeech_cache_snapshot(cache_dir: str | Path) -> Path:
    root = resolve_repo_path(cache_dir)
    candidates = sorted(root.glob("speechcolab___gigaspeech/xs/*/*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No cached GigaSpeech XS dataset_info.json found under {root}. "
            "Alpha will not download the dataset automatically."
        )
    return candidates[-1].parent


def cached_gigaspeech_info(cache_dir: str | Path) -> dict[str, Any]:
    snapshot = find_gigaspeech_cache_snapshot(cache_dir)
    return json.loads((snapshot / "dataset_info.json").read_text(encoding="utf-8"))


def load_cached_gigaspeech_split(cache_dir: str | Path, split: str):
    from datasets import Audio, Dataset, concatenate_datasets

    snapshot = find_gigaspeech_cache_snapshot(cache_dir)
    shards = sorted(snapshot.glob(f"gigaspeech-{split}-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"Cached GigaSpeech split '{split}' has no Arrow shards in {snapshot}")
    dataset = concatenate_datasets([Dataset.from_file(str(shard)) for shard in shards])
    dataset = dataset.select_columns(["audio", "text", "segment_id", "speaker", "begin_time", "end_time"])
    dataset = dataset.rename_column("segment_id", "source_id")
    dataset = dataset.rename_column("speaker", "speaker_id")
    dataset = dataset.map(
        lambda row: {
            "dataset": "gigaspeech",
            "split": split,
            "duration_seconds": max(0.0, float(row["end_time"]) - float(row["begin_time"])),
        },
        remove_columns=["begin_time", "end_time"],
        desc=f"Standardizing cached GigaSpeech {split}",
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def _resolve_local_audio(row: dict[str, Any], dataset_name: str, root_dir: str | Path | None) -> str:
    raw = Path(str(row.get("audio_path", "")).strip())
    if raw.is_absolute():
        return str(raw)
    if dataset_name == "readirect_letters" and root_dir:
        rooted = resolve_repo_path(root_dir) / raw
        if rooted.exists():
            return str(rooted)
    return str(resolve_repo_path(raw))


def load_local_manifest_dataset(
    manifest: str | Path,
    dataset_name: str,
    split: str,
    *,
    root_dir: str | Path | None = None,
):
    from datasets import Audio, Dataset

    normalized: list[dict[str, Any]] = []
    for row in read_jsonl(manifest):
        audio_path = _resolve_local_audio(row, dataset_name, root_dir)
        text = str(row.get("text") or row.get("letter") or "").strip()
        if not text or not Path(audio_path).exists():
            continue
        normalized.append(
            {
                "audio": audio_path,
                "text": text,
                "dataset": dataset_name,
                "split": split,
                "source_id": str(row.get("source_id") or row.get("original_audio_path") or row.get("audio_path") or ""),
                "speaker_id": str(row.get("speaker_id", "")),
                "duration_seconds": row.get("duration_seconds"),
            }
        )
    dataset = Dataset.from_list(normalized)
    if not normalized:
        return dataset
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def _cap_dataset(dataset: Any, maximum: int | None, seed: int) -> Any:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(maximum))


def _duration_filter(dataset: Any, minimum: float, maximum: float) -> Any:
    def keep(duration: Any) -> bool:
        if duration is None:
            return True
        try:
            value = float(duration)
        except (TypeError, ValueError):
            return False
        return minimum <= value <= maximum

    return dataset.filter(
        keep,
        input_columns=["duration_seconds"],
        desc=f"Filtering audio duration to {minimum}-{maximum}s",
    )


def build_alpha_raw_dataset(config: dict[str, Any], split: str):
    from datasets import concatenate_datasets

    data_cfg = config["data"]
    seed = int(config.get("run", {}).get("seed", 42))
    giga_cfg = data_cfg["gigaspeech"]
    speech_cfg = data_cfg["speechocean"]
    letters_cfg = data_cfg["readirect_letters"]
    split_key = "validation" if split == "validation" else split

    giga_split = str(giga_cfg[f"{split_key}_split"])
    gigaspeech = load_cached_gigaspeech_split(giga_cfg["cache_dir"], giga_split)
    speechocean = load_local_manifest_dataset(
        speech_cfg[f"{split_key}_manifest"], "speechocean", split
    )
    letters = load_local_manifest_dataset(
        letters_cfg[f"{split_key}_manifest"],
        "readirect_letters",
        split,
        root_dir=letters_cfg["root_dir"],
    )

    if split == "train":
        gigaspeech = _cap_dataset(gigaspeech, giga_cfg.get("max_train_samples"), seed)
        speechocean = _cap_dataset(speechocean, speech_cfg.get("max_train_samples"), seed + 1)
        letters = _cap_dataset(letters, letters_cfg.get("max_train_samples"), seed + 2)

    minimum = float(data_cfg.get("min_duration_seconds", 0.2))
    maximum = float(data_cfg.get("max_duration_seconds", 30.0))
    datasets = [
        _duration_filter(source, minimum, maximum)
        for source in (gigaspeech, speechocean, letters)
        if len(source)
    ]
    combined = concatenate_datasets(datasets)
    if bool(data_cfg.get("sampling", {}).get("shuffle", True)):
        combined = combined.shuffle(seed=seed)
    return combined


def dataset_distribution(dataset: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in dataset["dataset"]).items()))


def prepare_alpha_dataset(dataset: Any, processor: Any, config: dict[str, Any]):
    vocab = set(processor.tokenizer.get_vocab().keys())

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row["audio"]
        text = normalize_asr_text(row["text"], vocab)
        if not text:
            raise ValueError(f"Transcript became empty after normalization: {row.get('source_id')}")
        input_values = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        labels = processor.tokenizer(text).input_ids
        return {"input_values": input_values, "labels": labels}

    metadata_columns = ["dataset", "split", "source_id", "speaker_id", "text"]
    prepared = dataset.map(
        prepare,
        remove_columns=[column for column in dataset.column_names if column not in metadata_columns],
        num_proc=int(config["data"].get("preprocessing_num_proc", 1)),
        desc=f"Decoding and preprocessing {config.get('run', {}).get('name', 'ASR')} audio",
    )
    return prepared


def deterministic_rows(rows: list[dict[str, Any]], maximum: int | None, seed: int) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(rows):
        return rows
    rng = random.Random(seed)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), maximum))]
