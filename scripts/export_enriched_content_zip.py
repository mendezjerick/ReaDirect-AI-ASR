from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REFUSED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".mp4", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".gguf"}


def export_zip(source_dir: Path, output: Path, include_reports: bool = False) -> list[str]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.name == ".gitkeep":
                continue
            if path.suffix.lower() in REFUSED_EXTENSIONS:
                continue
            if "reports" in path.parts and not include_reports:
                continue
            relative = path.relative_to(source_dir).as_posix()
            archive.write(path, relative)
            written.append(relative)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package enriched content CSVs for review/import.")
    parser.add_argument("--source-dir", default="content_bank_enriched/import_ready", type=Path)
    parser.add_argument("--output", default="content_bank_enriched/readirect-enriched-content.zip", type=Path)
    parser.add_argument("--include-reports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = export_zip(args.source_dir, args.output, args.include_reports)
    print(f"Wrote {args.output}")
    print("ZIP contents:")
    for item in written:
        print(f"- {item}")


if __name__ == "__main__":
    main()
