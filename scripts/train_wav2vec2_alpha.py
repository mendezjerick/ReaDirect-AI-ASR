from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import (
    build_alpha_raw_dataset,
    configure_windows_ffmpeg,
    dataset_distribution,
    load_alpha_config,
    prepare_alpha_dataset,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean ReaDirect Wav2Vec2 Alpha ASR model.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_alpha.yaml"))
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def resolve_fp16(value: Any, cuda_available: bool) -> bool:
    if isinstance(value, str) and value.lower() == "true_if_cuda_available":
        return cuda_available
    return bool(value) and cuda_available


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=True, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch


def main() -> int:
    args = parse_args()
    configure_windows_ffmpeg()
    config = load_alpha_config(args.config)

    import torch
    from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor

    cuda_available = bool(torch.cuda.is_available())
    if bool(config["training"].get("require_cuda", True)) and not cuda_available:
        raise RuntimeError("Alpha requires CUDA. Training was not started.")

    model_cfg = config["model"]
    base_model_path = resolve_repo_path(model_cfg["base_model_path"])
    output_model_path = resolve_repo_path(model_cfg["output_model_path"])
    checkpoint_dir = resolve_repo_path(model_cfg["checkpoint_dir"])
    log_dir = resolve_repo_path(model_cfg["log_dir"])
    report_dir = resolve_repo_path(model_cfg["report_dir"])
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"Local base checkpoint not found: {base_model_path}. "
            f"Expected an existing local copy of {model_cfg['base_checkpoint']}."
        )
    if base_model_path.resolve() == output_model_path.resolve():
        raise RuntimeError("Alpha output path must not overwrite the base checkpoint.")

    processor = Wav2Vec2Processor.from_pretrained(str(base_model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(base_model_path), local_files_only=True)
    if bool(config["training"].get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    raw_dataset = build_alpha_raw_dataset(config, "train")
    distribution = dataset_distribution(raw_dataset)
    print(f"Alpha training rows: {len(raw_dataset)}")
    print(f"Alpha dataset distribution: {distribution}")
    train_dataset = prepare_alpha_dataset(raw_dataset, processor, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_model_path.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(checkpoint_dir),
        "logging_dir": str(log_dir / "tensorboard"),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(train_cfg.get("learning_rate", 3e-5)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 500)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "logging_steps": int(train_cfg.get("logging_steps", 25)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "fp16": resolve_fp16(train_cfg.get("fp16", False), cuda_available),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "save_strategy": "steps",
        "report_to": ["tensorboard"],
        "seed": int(config["run"].get("seed", 42)),
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "no"
    training_args = TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorCTCWithPadding(processor),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = processor
    trainer = Trainer(**trainer_kwargs)

    started_at = datetime.now(timezone.utc)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_model_path))
    processor.save_pretrained(str(output_model_path))
    completed_at = datetime.now(timezone.utc)
    metadata = {
        "run_name": "Alpha",
        "base_checkpoint": model_cfg["base_checkpoint"],
        "base_model_path": model_cfg["base_model_path"],
        "output_model_path": model_cfg["output_model_path"],
        "training_type": "clean_raw_asr_fine_tuning",
        "datasets": distribution,
        "dataset_mix": {key: value / len(raw_dataset) for key, value in distribution.items()},
        "excluded_features": ["beam_search", "expected_centric_correction", "boundary_repair", "gop_scoring"],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "training_metrics": dict(result.metrics),
        "evaluation_completed": False,
    }
    (output_model_path / "readirect_alpha_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Alpha model saved to {output_model_path}")
    print("Evaluation was not run. Use scripts/evaluate_wav2vec2_alpha.py separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

