from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from training.wav2vec2_alpha_data import (
    _duration_filter,
    _expand_env_value,
    load_local_manifest_dataset,
    prepare_alpha_dataset,
)
from training.wav2vec2_beta_data import gigaspeech_s_parquet_files
from training.wav2vec2_manifest_utils import resolve_repo_path


def load_gamma_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Gamma config not found: {config_path}")
    return _expand_env_value(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def configured_row_limit(value: Any) -> int | None:
    if value is None or value == "":
        return None
    limit = int(value)
    return None if limit == 0 else limit


def load_gamma_gigaspeech_train(
    config: dict[str, Any],
    *,
    max_samples: int | None = None,
    seed: int | None = None,
):
    from datasets import Audio, load_dataset

    files = gigaspeech_s_parquet_files(config)
    if not files:
        raise FileNotFoundError(
            f"No local GigaSpeech S parquet files found under "
            f"{resolve_repo_path(config['data']['gigaspeech']['parquet_dir'])}"
        )
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in files]},
        split="train",
        cache_dir=str(resolve_repo_path(config["data"]["gigaspeech"]["cache_dir"])),
    )
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed or int(config["run"].get("seed", 44))).select(
            range(max_samples)
        )
    dataset = dataset.select_columns(["audio", "text", "segment_id", "speaker", "begin_time", "end_time"])
    dataset = dataset.rename_column("segment_id", "source_id")
    dataset = dataset.rename_column("speaker", "speaker_id")
    dataset = dataset.map(
        lambda row: {
            "dataset": "gigaspeech",
            "split": "train",
            "duration_seconds": max(0.0, float(row["end_time"]) - float(row["begin_time"])),
        },
        remove_columns=["begin_time", "end_time"],
        desc="Standardizing full local GigaSpeech S train data",
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def build_gamma_train_dataset(config: dict[str, Any]):
    from datasets import concatenate_datasets

    data_cfg = config["data"]
    seed = int(config["run"].get("seed", 44))
    giga_cfg = data_cfg["gigaspeech"]
    letters_cfg = data_cfg["readirect_letters"]
    limit = configured_row_limit(giga_cfg.get("max_train_samples"))
    gigaspeech = load_gamma_gigaspeech_train(config, max_samples=limit, seed=seed)
    letters = load_local_manifest_dataset(
        letters_cfg["train_manifest"],
        "readirect_letters",
        "train",
        root_dir=letters_cfg["root_dir"],
    )
    repeat_factor = int(letters_cfg.get("repeat_factor", 1))
    if repeat_factor < 1:
        raise RuntimeError("READIRECT_LETTERS_REPEAT_FACTOR must be at least 1.")
    letter_sources = [letters] * repeat_factor

    minimum = float(data_cfg.get("min_duration_seconds", 0.2))
    maximum = float(data_cfg.get("max_duration_seconds", 30.0))
    sources = [_duration_filter(gigaspeech, minimum, maximum)]
    sources.extend(_duration_filter(source, minimum, maximum) for source in letter_sources)
    combined = concatenate_datasets(sources)
    if bool(data_cfg.get("sampling", {}).get("shuffle", True)):
        combined = combined.shuffle(seed=seed)
    return combined


def build_gamma_shared_dataset(config: dict[str, Any], split: str = "validation"):
    from datasets import concatenate_datasets

    split_key = "validation" if split == "validation" else split
    data_cfg = config["data"]
    speech_cfg = data_cfg["evaluation_speechocean"]
    letters_cfg = data_cfg["readirect_letters"]
    sources = [
        load_local_manifest_dataset(
            speech_cfg[f"{split_key}_manifest"], "speechocean", split
        ),
        load_local_manifest_dataset(
            letters_cfg[f"{split_key}_manifest"],
            "readirect_letters",
            split,
            root_dir=letters_cfg["root_dir"],
        ),
    ]
    minimum = float(data_cfg.get("min_duration_seconds", 0.2))
    maximum = float(data_cfg.get("max_duration_seconds", 30.0))
    return concatenate_datasets(
        [_duration_filter(source, minimum, maximum) for source in sources if len(source)]
    )


def load_cached_gigaspeech_evaluation(config: dict[str, Any], split: str = "validation"):
    from datasets import Audio, load_dataset

    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported GigaSpeech evaluation split: {split}")
    cache_dir = resolve_repo_path(config["data"]["gigaspeech"]["cache_dir"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = load_dataset(
        "speechcolab/gigaspeech",
        "xs",
        split=split,
        cache_dir=str(cache_dir),
        download_mode="reuse_dataset_if_exists",
    )
    dataset = dataset.select_columns(
        ["audio", "text", "segment_id", "speaker", "begin_time", "end_time"]
    )
    dataset = dataset.rename_column("segment_id", "source_id")
    dataset = dataset.rename_column("speaker", "speaker_id")
    dataset = dataset.map(
        lambda row: {
            "dataset": "gigaspeech",
            "split": split,
            "duration_seconds": max(
                0.0, float(row["end_time"]) - float(row["begin_time"])
            ),
        },
        remove_columns=["begin_time", "end_time"],
        desc=f"Standardizing cached GigaSpeech {split} data",
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))
    minimum = float(config["data"].get("min_duration_seconds", 0.2))
    maximum = float(config["data"].get("max_duration_seconds", 30.0))
    return _duration_filter(dataset, minimum, maximum)


def prepare_gamma_dataset(dataset: Any, processor: Any, config: dict[str, Any]):
    return prepare_alpha_dataset(dataset, processor, config)
