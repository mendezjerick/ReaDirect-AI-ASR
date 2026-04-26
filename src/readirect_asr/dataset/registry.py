from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_dataset_registry(path: str | Path = "configs/dataset_registry.yaml") -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in dataset registry: {registry_path}")
    return data


def list_active_datasets(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    active_registry = registry or load_dataset_registry()
    datasets = active_registry.get("datasets", {})
    return {
        name: config
        for name, config in datasets.items()
        if config.get("status") == "active"
    }


def list_planned_datasets(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    active_registry = registry or load_dataset_registry()
    datasets = active_registry.get("datasets", {})
    return {
        name: config
        for name, config in datasets.items()
        if str(config.get("status", "")).startswith("planned")
    }


def validate_dataset_paths(
    registry: dict[str, Any] | None = None,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    active_registry = registry or load_dataset_registry()
    root = Path(repo_root)
    missing: list[str] = []
    existing: list[str] = []
    for name, config in active_registry.get("datasets", {}).items():
        local_path = config.get("local_path")
        if not local_path:
            continue
        path = Path(local_path)
        candidate = path if path.is_absolute() else root / path
        if candidate.exists():
            existing.append(name)
        else:
            missing.append(name)
    return {"existing": existing, "missing": missing}
