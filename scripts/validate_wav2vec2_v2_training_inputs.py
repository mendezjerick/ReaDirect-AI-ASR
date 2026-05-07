from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


BASE_MODEL = Path("models/wav2vec2-readirect-asr")
OUTPUT_MODEL = Path("models/wav2vec2-readirect-asr-letters-v2")
READIRECT_ROOT = Path("external_datasets/readirect_letters")
READIRECT_MANIFESTS = {
    "train": READIRECT_ROOT / "manifests/readirect_letters_train.jsonl",
    "valid": READIRECT_ROOT / "manifests/readirect_letters_valid.jsonl",
    "test": READIRECT_ROOT / "manifests/readirect_letters_test.jsonl",
}
SPEECHOCEAN_MANIFESTS = {
    "train": Path("external_datasets/manifests/speechocean_train.jsonl"),
    "valid": Path("external_datasets/manifests/speechocean_valid.jsonl"),
    "test": Path("external_datasets/manifests/speechocean_test.jsonl"),
}
LIBRISPEECH_MANIFESTS = {
    "train": Path("external_datasets/manifests/librispeech_train_clean_100.jsonl"),
    "valid": Path("external_datasets/manifests/librispeech_dev_clean.jsonl"),
    "test": Path("external_datasets/manifests/librispeech_test_clean.jsonl"),
}


def exists(path: Path) -> bool:
    return resolve_repo_path(path).exists()


def resolve_audio_path(row: dict[str, Any], manifest_path: Path, dataset: str) -> Path:
    raw_path = Path(str(row.get("audio_path", "")).strip())
    if raw_path.is_absolute():
        return raw_path
    if dataset == "readirect_letters":
        candidate = resolve_repo_path(READIRECT_ROOT / raw_path)
        if candidate.exists():
            return candidate
    return resolve_repo_path(raw_path)


def validate_manifest_audio(manifest_path: Path, dataset: str, check_letters: bool) -> tuple[int, list[str]]:
    rows = read_jsonl(manifest_path)
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row, manifest_path, dataset)
        if not audio_path.exists():
            errors.append(f"{manifest_path}:{index} missing audio_path {row.get('audio_path')}")
        text = str(row.get("text") or row.get("letter") or "").strip()
        if check_letters and not re.fullmatch(r"[A-Z]", text):
            errors.append(f"{manifest_path}:{index} letter label must be single A-Z, got {text!r}")
        if not check_letters and not text:
            errors.append(f"{manifest_path}:{index} missing usable text label")
    return len(rows), errors


def cuda_status() -> tuple[bool, str]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if available else "CPU"
        return available, gpu_name
    except Exception as exc:
        return False, f"torch import failed: {exc}"


def check_path(label: str, path: Path, errors: list[str]) -> None:
    found = exists(path)
    print(f"{label}: {'FOUND' if found else 'MISSING'} - {path}")
    if not found:
        errors.append(f"Missing {label}: {path}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    check_path("Base model", BASE_MODEL, errors)
    output_separate = resolve_repo_path(BASE_MODEL).resolve() != resolve_repo_path(OUTPUT_MODEL).resolve()
    print(f"Output model path separate: {output_separate} - {OUTPUT_MODEL}")
    if not output_separate:
        errors.append("Output model path must be separate from the base model path.")

    check_path("Custom letter dataset", READIRECT_ROOT, errors)
    for split, path in READIRECT_MANIFESTS.items():
        check_path(f"ReaDirect letters {split} manifest", path, errors)
    for split, path in SPEECHOCEAN_MANIFESTS.items():
        check_path(f"SpeechOcean {split} manifest", path, errors)
    for split, path in LIBRISPEECH_MANIFESTS.items():
        check_path(f"LibriSpeech {split} manifest", path, errors)

    manifest_specs = [
        *[(path, "readirect_letters", True) for path in READIRECT_MANIFESTS.values()],
        *[(path, "speechocean", False) for path in SPEECHOCEAN_MANIFESTS.values()],
        *[(path, "librispeech", False) for path in LIBRISPEECH_MANIFESTS.values()],
    ]
    for path, dataset, check_letters in manifest_specs:
        if not exists(path):
            continue
        row_count, manifest_errors = validate_manifest_audio(path, dataset, check_letters)
        print(f"Manifest rows checked: {path} = {row_count}")
        if manifest_errors:
            errors.extend(manifest_errors[:25])
            if len(manifest_errors) > 25:
                warnings.append(f"{path} had {len(manifest_errors)} errors; showing first 25.")

    cuda_available, gpu_name = cuda_status()
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"GPU: {gpu_name}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidation passed. Inputs are ready for v2 training manifest construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
