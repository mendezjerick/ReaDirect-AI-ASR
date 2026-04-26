from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.content.validation import validate_content_bank


RECOGNIZED_FOLDERS = {"assessment", "modules", "agents", "rules", "feedback", "prompts", "docs"}
REFUSED_SUFFIXES = {
    ".env",
    ".sql",
    ".dump",
    ".backup",
    ".wav",
    ".mp3",
    ".m4a",
    ".webm",
    ".mp4",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
}


def _safe_to_copy(path: Path) -> bool:
    return path.suffix.lower() not in REFUSED_SUFFIXES and path.name != ".env"


def import_zip(zip_path: Path, dest: Path, overwrite: bool = False) -> list[Path]:
    copied: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_path)

        candidates = [path for path in tmp_path.rglob("*") if path.is_dir() and path.name in RECOGNIZED_FOLDERS]
        for folder in candidates:
            target_folder = dest / folder.name
            target_folder.mkdir(parents=True, exist_ok=True)
            for source in folder.rglob("*"):
                if source.is_dir():
                    continue
                if not _safe_to_copy(source):
                    print(f"Refused unsafe file: {source.name}")
                    continue
                relative = source.relative_to(folder)
                target = target_folder / relative
                if target.exists() and not overwrite:
                    print(f"Skipped existing file: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a ReaDirect content-bank ZIP export.")
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--dest", default="content_bank", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = import_zip(args.zip_path, args.dest, args.overwrite)
    print(f"Copied {len(copied)} files")
    for path in copied:
        print(f"- {path}")
    report = validate_content_bank(args.dest)
    print(f"Validation ok: {report['ok']}")
    print(f"Missing required files: {len(report['missing_required_files'])}")
    print(f"Column errors: {len(report['column_errors'])}")


if __name__ == "__main__":
    main()

