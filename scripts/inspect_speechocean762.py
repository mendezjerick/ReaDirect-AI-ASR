from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".csv", ".scp", ".ark", ".lab", ".TextGrid"}


def find_archive(dataset_dir: Path, archive: Path | None = None) -> Path | None:
    if archive and archive.exists():
        return archive
    default = dataset_dir / "raw" / "speechocean762.tar.gz"
    if default.exists():
        return default
    candidates = sorted((dataset_dir / "raw").glob("*.tar.gz")) if (dataset_dir / "raw").exists() else []
    return candidates[0] if candidates else None


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"]


def print_tree(root: Path, max_files: int) -> None:
    for index, path in enumerate(iter_files(root)):
        if index >= max_files:
            print(f"... truncated after {max_files} files")
            break
        print(path.relative_to(root).as_posix())


def inspect(dataset_dir: Path, archive: Path | None = None, max_files: int = 50) -> dict[str, object]:
    archive_path = find_archive(dataset_dir, archive)
    extracted_dir = dataset_dir / "extracted"
    extracted_files = iter_files(extracted_dir)
    audio_counts = Counter(path.suffix.lower() for path in extracted_files if path.suffix.lower() in AUDIO_EXTENSIONS)
    annotation_counts = Counter(path.suffix for path in extracted_files if path.suffix in ANNOTATION_EXTENSIONS)
    readmes = [path for path in extracted_files if path.name.lower() in {"readme.md", "readme.txt", "license", "license.txt"}]
    return {
        "archive_path": str(archive_path) if archive_path else "",
        "archive_exists": archive_path is not None and archive_path.exists(),
        "extracted_dir": str(extracted_dir),
        "extracted_file_count": len(extracted_files),
        "audio_counts": dict(audio_counts),
        "annotation_counts": dict(annotation_counts),
        "readme_license_files": [str(path) for path in readmes],
        "sample_files": [str(path.relative_to(extracted_dir)) for path in extracted_files[:max_files]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Speechocean762 archive or extracted folder.")
    parser.add_argument("--dataset-dir", default="external_datasets/speechocean762", type=Path)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--print-tree", action="store_true")
    parser.add_argument("--max-files", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect(args.dataset_dir, args.archive, args.max_files)
    print(f"Archive exists: {report['archive_exists']}")
    print(f"Archive path: {report['archive_path'] or 'not found'}")
    print(f"Extracted dir: {report['extracted_dir']}")
    print(f"Extracted file count: {report['extracted_file_count']}")
    print(f"Audio counts: {report['audio_counts']}")
    print(f"Annotation counts: {report['annotation_counts']}")
    print(f"README/license files: {len(report['readme_license_files'])}")
    if args.print_tree:
        print_tree(Path(report["extracted_dir"]), args.max_files)


if __name__ == "__main__":
    main()
