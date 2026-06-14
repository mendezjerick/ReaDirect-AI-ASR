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
from training.wav2vec2_delta_data import (
    build_delta_shared_dataset,
    build_delta_train_dataset,
    load_delta_config,
    prepare_delta_dataset,
)
from training.wav2vec2_manifest_utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue Beta into ReaDirect Wav2Vec2 Delta.")
    parser.add_argument("--config", type=Path, default=Path("configs/wav2vec2_delta.yaml"))
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
        labels = self.processor.pad(labels=label_features, padding=True, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch


def verify_beta_processor_and_head(beta_path: Path, base_path: Path) -> None:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    beta_processor = Wav2Vec2Processor.from_pretrained(str(beta_path), local_files_only=True)
    beta_model = Wav2Vec2ForCTC.from_pretrained(str(beta_path), local_files_only=True)
    base_processor = Wav2Vec2Processor.from_pretrained(str(base_path), local_files_only=True)
    base_model = Wav2Vec2ForCTC.from_pretrained(str(base_path), local_files_only=True)
    if beta_processor.tokenizer.get_vocab() != base_processor.tokenizer.get_vocab():
        raise RuntimeError("Beta tokenizer differs from base-960h. Delta training refused.")
    if beta_processor.tokenizer.word_delimiter_token_id != base_processor.tokenizer.word_delimiter_token_id:
        raise RuntimeError("Beta word delimiter differs from base-960h. Delta training refused.")
    if beta_model.lm_head.weight.shape != base_model.lm_head.weight.shape:
        raise RuntimeError("Beta CTC head shape differs from base-960h. Delta training refused.")
    if beta_processor.feature_extractor.sampling_rate != 16000:
        raise RuntimeError("Beta processor is not configured for 16 kHz.")
    print("Safety check passed: Delta preserves Beta's tokenizer, processor, and CTC head.")


def build_compute_metrics(processor: Any):
    import jiwer

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return {
            "wer": float(jiwer.wer(references, predictions)),
            "cer": float(jiwer.cer(references, predictions)),
        }

    return compute_metrics


def main() -> int:
    args = parse_args()
    configure_windows_ffmpeg()
    config = load_delta_config(args.config)

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    cuda_available = bool(torch.cuda.is_available())
    if bool(config["training"].get("require_cuda", True)) and not cuda_available:
        raise RuntimeError("Delta requires CUDA. Training was not started.")

    model_cfg = config["model"]
    beta_path = resolve_repo_path(model_cfg["beta_checkpoint_path"])
    base_path = resolve_repo_path(model_cfg["reference_base_model_path"])
    output_path = resolve_repo_path(model_cfg["output_model_path"])
    checkpoint_dir = resolve_repo_path(model_cfg["checkpoint_dir"])
    report_dir = resolve_repo_path(model_cfg["report_dir"])
    log_dir = resolve_repo_path(model_cfg["log_dir"])
    required = ("config.json", "model.safetensors", "vocab.json", "processor_config.json")
    missing = [name for name in required if not (beta_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Beta model is incomplete at {beta_path}; missing {missing}")
    if beta_path.resolve() == output_path.resolve():
        raise RuntimeError("Delta output cannot overwrite Beta.")
    verify_beta_processor_and_head(beta_path, base_path)

    train_cfg = config["training"]
    epochs = float(train_cfg.get("num_train_epochs", 1))
    maximum_epochs = float(train_cfg.get("max_allowed_epochs", 2))
    if not 0 < epochs <= maximum_epochs:
        raise RuntimeError(f"Delta epochs must be greater than 0 and no more than {maximum_epochs}.")
    if float(train_cfg.get("learning_rate", 5e-6)) >= 1e-5:
        raise RuntimeError("Delta learning rate must remain lower than Beta's 1e-5.")

    print(f"Delta training start checkpoint: {beta_path}")
    processor = Wav2Vec2Processor.from_pretrained(str(beta_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(beta_path), local_files_only=True)
    if bool(train_cfg.get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()
    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    raw_train, mix_summary = build_delta_train_dataset(config, write_summary=True)
    raw_valid = build_delta_shared_dataset(config, "validation", include_gigaspeech=False)
    distribution = dataset_distribution(raw_train)
    if "speechocean" in distribution:
        raise RuntimeError("SpeechOcean appeared in Delta training data. Training refused.")
    print(f"Delta training rows: {len(raw_train)}")
    print(f"Delta effective distribution: {distribution}")
    print(f"Delta shared validation rows: {len(raw_valid)}")
    train_dataset = prepare_delta_dataset(raw_train, processor, config)
    eval_dataset = prepare_delta_dataset(raw_valid, processor, config)

    for directory in (output_path, checkpoint_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(checkpoint_dir),
        "logging_dir": str(log_dir / "tensorboard"),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(train_cfg.get("learning_rate", 5e-6)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(train_cfg.get("logging_steps", 25)),
        "eval_steps": int(train_cfg.get("eval_steps", 500)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "num_train_epochs": epochs,
        "fp16": resolve_fp16(train_cfg.get("fp16", False), cuda_available),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "report_to": ["tensorboard"],
        "seed": int(config["run"].get("seed", 45)),
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    training_args = TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorCTCWithPadding(processor),
        "compute_metrics": build_compute_metrics(processor),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2)),
                early_stopping_threshold=float(train_cfg.get("early_stopping_threshold", 0.0)),
            )
        ],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = processor
    trainer = Trainer(**trainer_kwargs)

    started_at = datetime.now(timezone.utc)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_path))
    processor.save_pretrained(str(output_path))
    metadata = {
        "run_name": "Delta",
        "start_checkpoint": str(beta_path),
        "best_delta_checkpoint": trainer.state.best_model_checkpoint,
        "output_model_path": str(output_path),
        "dataset_mix": mix_summary,
        "best_shared_validation_wer": trainer.state.best_metric,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "training_metrics": dict(result.metrics),
        "evaluation_completed": False,
    }
    (output_path / "readirect_delta_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Delta best model saved to {output_path}")
    print(f"Best checkpoint selected by shared validation WER: {trainer.state.best_model_checkpoint}")
    print("Full Delta evaluation was not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

