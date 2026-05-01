from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}
MANIFEST_FIELDS = (
    "audio_path",
    "text",
    "dataset",
    "split",
    "speaker_id",
    "duration_seconds",
    "sample_rate",
    "source_id",
    "metadata",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def audio_info(path: str | Path) -> tuple[float | None, int | None]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = round(float(info.frames) / float(info.samplerate), 6) if info.samplerate else None
        return duration, int(info.samplerate) if info.samplerate else None
    except Exception:
        return None, None


def iter_audio_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = resolve_repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = resolve_repo_path(path)
    rows: list[dict[str, Any]] = []
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row.setdefault("_line_number", line_number)
            rows.append(row)
    return rows


def make_manifest_row(
    *,
    audio_path: str | Path,
    text: str,
    dataset: str,
    split: str,
    speaker_id: str = "",
    source_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration, sample_rate = audio_info(audio_path)
    return {
        "audio_path": str(audio_path),
        "text": text,
        "dataset": dataset,
        "split": split,
        "speaker_id": speaker_id,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "source_id": source_id,
        "metadata": metadata or {},
    }


def deterministic_sample(rows: list[dict[str, Any]], size: int, seed: int = 42) -> list[dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def split_train_valid(rows: list[dict[str, Any]], valid_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        return rows, []
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    return train, valid

