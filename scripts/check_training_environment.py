from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    torch = _try_import("torch")
    if torch:
        cuda_available = bool(torch.cuda.is_available())
        print(f"Torch: {getattr(torch, '__version__', 'unknown')}")
        print(f"CUDA available: {cuda_available}")
        print(f"Torch CUDA version: {torch.version.cuda}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            try:
                props = torch.cuda.get_device_properties(0)
                print(f"VRAM: {round(props.total_memory / (1024 ** 3), 2)} GB")
            except Exception:
                print("VRAM: unavailable")
        else:
            print("GPU: CPU only")
            print("PyTorch may be CPU-only. Install CUDA-enabled PyTorch using the official PyTorch selector, then restart PowerShell.")
    else:
        print("Torch: not installed")
    for package in ("transformers", "datasets", "evaluate", "accelerate"):
        module = _try_import(package)
        print(f"{package}: {getattr(module, '__version__', 'not installed') if module else 'not installed'}")
    for path in (
        "data/processed/whisper_finetune/train.jsonl",
        "data/processed/whisper_finetune/validation.jsonl",
        "data/processed/whisper_finetune/test.jsonl",
    ):
        print(f"{path}: {'exists' if Path(path).exists() else 'missing'}")
    print("Recommended RTX 3060 12GB settings: batch size 2, gradient accumulation 8, fp16 true, gradient checkpointing true.")
    return 0


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
