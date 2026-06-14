from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import (
    cached_gigaspeech_info,
    configure_windows_ffmpeg,
    find_gigaspeech_cache_snapshot,
    load_alpha_config,
)
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Alpha without starting training.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_alpha.yaml"))
    parser.add_argument("--decode-sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        ffmpeg_dir = configure_windows_ffmpeg()
    except Exception as exc:
        ffmpeg_dir = None
        errors.append(str(exc))
    config = load_alpha_config(args.config)

    base_model = resolve_repo_path(config["model"]["base_model_path"])
    if not (base_model / "config.json").exists():
        errors.append(f"Base model is missing or incomplete: {base_model}")

    cache_dir = config["data"]["gigaspeech"]["cache_dir"]
    try:
        snapshot = find_gigaspeech_cache_snapshot(cache_dir)
        info = cached_gigaspeech_info(cache_dir)
        split_counts = {
            name: details["num_examples"] for name, details in info.get("splits", {}).items()
        }
        print(f"GigaSpeech cache snapshot: {snapshot}")
        print(f"GigaSpeech cached split counts: {json.dumps(split_counts, sort_keys=True)}")
    except Exception as exc:
        errors.append(str(exc))

    for source_name in ("speechocean", "readirect_letters"):
        source = config["data"][source_name]
        for split in ("train", "validation", "test"):
            manifest = resolve_repo_path(source[f"{split}_manifest"])
            rows = read_jsonl(manifest)
            if not rows:
                errors.append(f"{source_name} {split} manifest is missing or empty: {manifest}")
            else:
                print(f"{source_name} {split}: {len(rows)} rows")

    if importlib.util.find_spec("torchcodec") is None:
        errors.append("TorchCodec is not installed. Install a version compatible with the installed PyTorch.")
    else:
        try:
            print(f"TorchCodec package version: {importlib.metadata.version('torchcodec')}")
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import torchcodec; print('TorchCodec import OK')"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if probe.returncode:
                detail = (probe.stderr or probe.stdout).strip()
                if len(detail) > 1600:
                    detail = detail[:1600] + "\n... output truncated ..."
                errors.append(f"TorchCodec cannot load: {detail}")
            else:
                print(probe.stdout.strip())
        except subprocess.TimeoutExpired:
            errors.append("TorchCodec import timed out after 20 seconds; FFmpeg DLL loading is not healthy.")

    ffmpeg_executable = shutil.which("ffmpeg")
    dll_candidates = []
    search_dirs = [Path(ffmpeg_dir)] if ffmpeg_dir else []
    search_dirs.extend(Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)
    for directory in search_dirs:
        dll_candidates.extend(directory.glob("avcodec-*.dll"))
    if not ffmpeg_executable:
        warnings.append("ffmpeg.exe is not on PATH.")
    if os.name == "nt" and not dll_candidates:
        errors.append(
            "FFmpeg shared libraries (for example avcodec-*.dll) were not found. "
            "Set FFMPEG_BIN_DIR to the FFmpeg shared-build bin directory."
        )

    if args.decode_sample and not errors:
        from training.wav2vec2_alpha_data import load_cached_gigaspeech_split

        dataset = load_cached_gigaspeech_split(cache_dir, config["data"]["gigaspeech"]["train_split"])
        audio = dataset[0]["audio"]
        print(
            f"Decoded GigaSpeech sample: samples={len(audio['array'])}, "
            f"sample_rate={audio['sampling_rate']}"
        )
    elif not args.decode_sample:
        warnings.append("Audio decoding was not tested. Re-run with --decode-sample before training.")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("Alpha setup validation failed. Training was not started.")
        return 1
    print("Alpha setup validation passed. Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
