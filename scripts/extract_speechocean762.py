from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".csv", ".scp", ".ark", ".lab", ".TextGrid"}


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_members(archive: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    safe: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target = dest / member.name
        if not _is_within_directory(dest, target):
            raise ValueError(f"Unsafe tar member path rejected: {member.name}")
        safe.append(member)
    return safe


def extract_archive(archive_path: Path, dest: Path, force: bool = False, dry_run: bool = False) -> dict[str, object]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    existing_files = [path for path in dest.rglob("*") if path.is_file() and path.name != ".gitkeep"] if dest.exists() else []
    if existing_files and not force and not dry_run:
        raise FileExistsError(f"Destination already contains files: {dest}. Use --force to overwrite/extract anyway.")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = safe_members(archive, dest)
        file_members = [member for member in members if member.isfile()]
        audio_count = sum(Path(member.name).suffix.lower() in AUDIO_EXTENSIONS for member in file_members)
        annotation_count = sum(Path(member.name).suffix in ANNOTATION_EXTENSIONS for member in file_members)
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            archive.extractall(dest, members=members)
    return {
        "extracted_root": str(dest),
        "file_count": len(file_members),
        "audio_count": audio_count,
        "annotation_count": annotation_count,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely extract Speechocean762 tar.gz.")
    parser.add_argument("--archive", default="external_datasets/speechocean762/raw/speechocean762.tar.gz", type=Path)
    parser.add_argument("--dest", default="external_datasets/speechocean762/extracted", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = extract_archive(args.archive, args.dest, args.force, args.dry_run)
    print(f"Extracted root: {report['extracted_root']}")
    print(f"File count: {report['file_count']}")
    print(f"Audio count: {report['audio_count']}")
    print(f"Annotation count: {report['annotation_count']}")
    print(f"Dry run: {report['dry_run']}")


if __name__ == "__main__":
    main()
