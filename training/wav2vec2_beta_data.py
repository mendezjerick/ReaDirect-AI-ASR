from __future__ import annotations

from pathlib import Path
from typing import Any

from training.wav2vec2_alpha_data import (
    _cap_dataset,
    _duration_filter,
    _expand_env_value,
    load_local_manifest_dataset,
    prepare_alpha_dataset,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def load_beta_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Beta config not found: {config_path}")
    return _expand_env_value(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def gigaspeech_s_parquet_files(config: dict[str, Any]) -> list[Path]:
    giga_cfg = config["data"]["gigaspeech"]
    parquet_dir = resolve_repo_path(giga_cfg["parquet_dir"])
    return sorted(parquet_dir.glob(str(giga_cfg.get("train_glob", "train-*.parquet"))))


def load_gigaspeech_s_train(
    config: dict[str, Any],
    *,
    max_samples: int | None = None,
    seed: int | None = None,
):
    from datasets import Audio, load_dataset

    files = gigaspeech_s_parquet_files(config)
    if not files:
        raise FileNotFoundError(
            f"No downloaded GigaSpeech S parquet files found under "
            f"{resolve_repo_path(config['data']['gigaspeech']['parquet_dir'])}"
        )
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in files]},
        split="train",
        cache_dir=str(resolve_repo_path(config["data"]["gigaspeech"]["cache_dir"])),
    )
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed or int(config["run"].get("seed", 43))).select(
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
        desc="Standardizing local GigaSpeech S train data",
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000, num_channels=1))


def build_beta_train_dataset(config: dict[str, Any]):
    from datasets import concatenate_datasets

    data_cfg = config["data"]
    seed = int(config["run"].get("seed", 43))
    giga_cfg = data_cfg["gigaspeech"]
    speech_cfg = data_cfg["speechocean"]
    letters_cfg = data_cfg["readirect_letters"]

    gigaspeech = load_gigaspeech_s_train(
        config,
        max_samples=giga_cfg.get("max_train_samples"),
        seed=seed,
    )
    speechocean = _cap_dataset(
        load_local_manifest_dataset(speech_cfg["train_manifest"], "speechocean", "train"),
        speech_cfg.get("max_train_samples"),
        seed + 1,
    )
    letters = _cap_dataset(
        load_local_manifest_dataset(
            letters_cfg["train_manifest"],
            "readirect_letters",
            "train",
            root_dir=letters_cfg["root_dir"],
        ),
        letters_cfg.get("max_train_samples"),
        seed + 2,
    )
    minimum = float(data_cfg.get("min_duration_seconds", 0.2))
    maximum = float(data_cfg.get("max_duration_seconds", 30.0))
    combined = concatenate_datasets(
        [_duration_filter(source, minimum, maximum) for source in (gigaspeech, speechocean, letters)]
    )
    if bool(data_cfg.get("sampling", {}).get("shuffle", True)):
        combined = combined.shuffle(seed=seed)
    return combined


def build_beta_shared_dataset(config: dict[str, Any], split: str = "validation"):
    from datasets import concatenate_datasets

    split_key = "validation" if split == "validation" else split
    data_cfg = config["data"]
    speech_cfg = data_cfg["speechocean"]
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


def prepare_beta_dataset(dataset: Any, processor: Any, config: dict[str, Any]):
    return prepare_alpha_dataset(dataset, processor, config)
