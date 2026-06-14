from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_alpha_data import configure_windows_ffmpeg, dataset_distribution
from training.wav2vec2_epsilon_data import (
    build_epsilon_shared_dataset,
    build_epsilon_train_dataset,
    load_epsilon_config,
    prepare_epsilon_dataset,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Delta into controlled Epsilon.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_epsilon.yaml"))
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{"input_values": feature["input_values"]} for feature in features]
        labels = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
        padded = self.processor.pad(labels=labels, padding=True, return_tensors="pt")
        batch["labels"] = padded["input_ids"].masked_fill(
            padded.attention_mask.ne(1), -100
        )
        return batch


def verify_delta_processor_and_head(delta_path: Path, base_path: Path) -> None:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    delta_processor = Wav2Vec2Processor.from_pretrained(
        str(delta_path), local_files_only=True
    )
    delta_model = Wav2Vec2ForCTC.from_pretrained(str(delta_path), local_files_only=True)
    base_processor = Wav2Vec2Processor.from_pretrained(
        str(base_path), local_files_only=True
    )
    base_model = Wav2Vec2ForCTC.from_pretrained(str(base_path), local_files_only=True)
    if delta_processor.tokenizer.get_vocab() != base_processor.tokenizer.get_vocab():
        raise RuntimeError("Delta tokenizer differs from base-960h. Epsilon refused.")
    if delta_model.lm_head.weight.shape != base_model.lm_head.weight.shape:
        raise RuntimeError("Delta CTC head shape differs from base-960h. Epsilon refused.")
    if delta_processor.feature_extractor.sampling_rate != 16000:
        raise RuntimeError("Delta processor is not configured for 16 kHz.")
    print("Safety check passed: Epsilon preserves Delta's tokenizer and CTC head.")


def build_compute_metrics(processor: Any, source_names: list[str]):
    import jiwer

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits = (
            prediction.predictions[0]
            if isinstance(prediction.predictions, tuple)
            else prediction.predictions
        )
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        letter_indices = [
            index for index, source in enumerate(source_names)
            if source == "readirect_letters"
        ]
        letter_accuracy = sum(
            references[index] == predictions[index] for index in letter_indices
        ) / len(letter_indices)
        return {
            "wer": float(jiwer.wer(references, predictions)),
            "cer": float(jiwer.cer(references, predictions)),
            "letter_accuracy": float(letter_accuracy),
        }

    return compute_metrics


def main() -> int:
    args = parse_args()
    configure_windows_ffmpeg()
    config = load_epsilon_config(args.config)

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    cuda_available = bool(torch.cuda.is_available())
    if config["training"].get("require_cuda", True) and not cuda_available:
        raise RuntimeError("Epsilon requires CUDA. Training was not started.")
    model_cfg = config["model"]
    delta_path = resolve_repo_path(model_cfg["delta_checkpoint_path"])
    base_path = resolve_repo_path(model_cfg["reference_base_model_path"])
    output_path = resolve_repo_path(model_cfg["output_model_path"])
    checkpoint_dir = resolve_repo_path(model_cfg["checkpoint_dir"])
    report_dir = resolve_repo_path(model_cfg["report_dir"])
    log_dir = resolve_repo_path(model_cfg["log_dir"])
    if output_path.resolve() in {
        resolve_repo_path("models/asr/beta").resolve(),
        resolve_repo_path("models/asr/gamma").resolve(),
        resolve_repo_path("models/asr/delta").resolve(),
    }:
        raise RuntimeError("Epsilon output would overwrite an earlier model.")
    required = ("config.json", "model.safetensors", "vocab.json", "processor_config.json")
    missing = [name for name in required if not (delta_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Delta model is incomplete at {delta_path}: {missing}")
    verify_delta_processor_and_head(delta_path, base_path)

    train_cfg = config["training"]
    epochs = float(args.epochs or train_cfg.get("num_train_epochs", 1))
    learning_rate = float(args.learning_rate or train_cfg.get("learning_rate", 5e-6))
    if not 0 < epochs <= float(train_cfg.get("max_allowed_epochs", 2)):
        raise RuntimeError("Epsilon epochs must be greater than 0 and no more than 2.")
    if not 0 < learning_rate <= 5e-6:
        raise RuntimeError("Epsilon learning rate must be positive and no greater than 5e-6.")

    print(f"Epsilon training start checkpoint: {delta_path}")
    print(f"Epsilon epochs: {epochs}; learning rate: {learning_rate}")
    processor = Wav2Vec2Processor.from_pretrained(str(delta_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(delta_path), local_files_only=True)
    if train_cfg.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    vocab = set(processor.tokenizer.get_vocab())
    raw_train, mix_summary = build_epsilon_train_dataset(config, vocab)
    raw_valid = build_epsilon_shared_dataset(config, "validation")
    distribution = dataset_distribution(raw_train)
    expected = {
        "gigaspeech": 18000,
        "slr83_southern_english": 12000,
        "speechocean": 6000,
        "readirect_letters": 4000,
    }
    if distribution != expected:
        raise RuntimeError(f"Unexpected Epsilon effective mix: {distribution}")
    print(f"Epsilon effective training distribution: {distribution}")
    print("SLR83 held-out evaluation rows in training: 0")
    source_names = list(raw_valid["dataset"])
    train_dataset = prepare_epsilon_dataset(raw_train, processor, config)
    eval_dataset = prepare_epsilon_dataset(raw_valid, processor, config)

    for directory in (output_path, checkpoint_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(checkpoint_dir),
        "logging_dir": str(log_dir / "tensorboard"),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 2)),
        "learning_rate": learning_rate,
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(train_cfg.get("logging_steps", 25)),
        "eval_steps": int(train_cfg.get("eval_steps", 250)),
        "save_steps": int(train_cfg.get("save_steps", 250)),
        "num_train_epochs": epochs,
        "fp16": cuda_available,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "save_total_limit": 1,
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "report_to": ["tensorboard"],
        "seed": int(config["run"].get("seed", 47)),
    }
    kwargs["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "steps"
    training_args = TrainingArguments(
        **{key: value for key, value in kwargs.items() if key in parameters}
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorCTCWithPadding(processor),
        "compute_metrics": build_compute_metrics(processor, source_names),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2)),
                early_stopping_threshold=float(train_cfg.get("early_stopping_threshold", 0.0)),
            )
        ],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs[
        "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    ] = processor
    trainer = Trainer(**trainer_kwargs)

    started_at = datetime.now(timezone.utc)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_path))
    processor.save_pretrained(str(output_path))
    metadata = {
        "run_name": "Epsilon",
        "start_checkpoint": str(delta_path),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "output_model_path": str(output_path),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dataset_mix": mix_summary,
        "best_shared_validation_wer": trainer.state.best_metric,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "training_metrics": dict(result.metrics),
        "heldout_slr83_used_for_training_or_early_stopping": False,
        "evaluation_completed": False,
    }
    (output_path / "readirect_epsilon_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Epsilon best model saved to {output_path}")
    print("Full Epsilon evaluation was not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
