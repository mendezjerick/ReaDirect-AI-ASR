from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = {
    "torch": "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "torchaudio": "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "transformers": "pip install transformers",
    "datasets": "pip install datasets",
    "soundfile": "pip install soundfile",
    "librosa": "pip install librosa",
    "pandas": "pip install pandas",
    "numpy": "pip install numpy",
    "accelerate": "pip install accelerate",
    "yaml": "pip install PyYAML",
}
OPTIONAL_EVAL_PACKAGES = {
    "evaluate": "pip install evaluate",
    "jiwer": "pip install jiwer",
    "sklearn": "pip install scikit-learn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local Wav2Vec2 fine-tuning setup.")
    parser.add_argument("--stage", choices=("mixed", "librispeech", "speechocean"), default="mixed")
    return parser.parse_args()


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        return True, str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return False, str(exc)


def path_exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).exists()


def first_existing(paths: list[str]) -> str | None:
    return next((path for path in paths if path_exists(path)), None)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    for package, install_cmd in REQUIRED_PACKAGES.items():
        ok, detail = import_status(package)
        print(f"{package}: {detail if ok else 'missing'}")
        if not ok:
            errors.append(f"Missing required package '{package}'. Install with: {install_cmd}")

    for package, install_cmd in OPTIONAL_EVAL_PACKAGES.items():
        ok, detail = import_status(package)
        print(f"{package} (optional evaluation): {detail if ok else 'missing'}")
        if not ok:
            warnings.append(f"Missing optional evaluation package '{package}'. Install with: {install_cmd}")

    torch_ok, _ = import_status("torch")
    if torch_ok:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Torch CUDA version: {torch.version.cuda}")
        else:
            warnings.append("CUDA is not available. Training will fall back to CPU and will be slow.")

    if not path_exists("models/wav2vec2-base-960h"):
        errors.append("Missing local ASR base model: models/wav2vec2-base-960h")
    else:
        print("ASR base model: models/wav2vec2-base-960h")

    if not path_exists("models/wav2vec2-phoneme"):
        warnings.append("Missing phoneme support model: models/wav2vec2-phoneme")
    else:
        print("Phoneme support model: models/wav2vec2-phoneme")

    librispeech_train = first_existing(
        [
            "external_datasets/librispeech/extracted/LibriSpeech/train-clean-100",
            "external_datasets/LibriSpeech/extracted/LibriSpeech/train-clean-100",
            "external_datasets/LibriSpeech/train-clean-100",
        ]
    )
    speechocean_root = first_existing(
        [
            "external_datasets/speechocean/extracted",
            "external_datasets/speechocean762/extracted",
        ]
    )
    print(f"LibriSpeech train-clean-100: {librispeech_train or 'missing'}")
    print(f"SpeechOcean extracted folder: {speechocean_root or 'missing'}")

    if args.stage == "librispeech" and not librispeech_train:
        errors.append("Selected stage requires LibriSpeech train-clean-100, but it was not found.")
    elif not librispeech_train:
        warnings.append("LibriSpeech train-clean-100 not found. Mixed training can only run if another train manifest exists.")

    if args.stage == "speechocean" and not speechocean_root:
        errors.append("Selected stage requires SpeechOcean, but no extracted SpeechOcean folder was found.")
    elif not speechocean_root:
        warnings.append("SpeechOcean not found. LibriSpeech-only basic training can still proceed if its manifest exists.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 2
    print("\nTraining setup validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

