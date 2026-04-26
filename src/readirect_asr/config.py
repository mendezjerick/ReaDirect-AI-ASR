from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    asr_provider: str = "mock"
    asr_model_size: str = "base.en"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = "en"
    content_bank_path: str = "content_bank"
    data_manifest_path: str = "data/manifests/dataset_manifest.csv"
    audio_base_path: str = "data/raw"
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    log_level: str = "info"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return data


def load_env() -> None:
    load_dotenv()

