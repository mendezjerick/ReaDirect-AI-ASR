from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.finetuning.whisper_dataset import summarize_whisper_dataset, validate_whisper_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded Whisper fine-tuning runner.")
    parser.add_argument("--config", default="configs/whisper_finetune_config.yaml", type=Path)
    parser.add_argument("--run", action="store_true", help="Actually start training. Required for any model download/training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data and exit without training.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    print_training_summary(config, args)
    validation = validate_training_inputs(config)
    if validation["errors"]:
        print("Training input errors:")
        for error in validation["errors"]:
            print(f"- {error}")
        return 2
    print(f"Dataset summary: {validation['summary']}")
    cuda = check_cuda()
    print(f"CUDA available: {cuda['available']}")
    print(f"GPU: {cuda['gpu_name']}")
    print(f"Torch CUDA version: {cuda['cuda_version']}")
    if not args.run or args.dry_run:
        if bool(config.get("runtime", {}).get("require_cuda", True)) and not cuda["available"]:
            print("Dry-run warning: CUDA is required for training config but is not currently available.")
        print("Dry run complete. No model download or training started. Pass --run to train manually.")
        return 0
    require_cuda = bool(config.get("runtime", {}).get("require_cuda", True) or args.require_cuda)
    allow_cpu = bool(config.get("runtime", {}).get("allow_cpu_training", False) or args.allow_cpu)
    if require_cuda and not cuda["available"]:
        print("CUDA is required by config but is not available. Install CUDA-enabled PyTorch before training.")
        return 3
    if not cuda["available"] and not allow_cpu:
        print("CPU training is disabled. Pass --allow-cpu only for tiny debugging runs.")
        return 3
    run_training(config, args)
    return 0


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config.setdefault("model", {})
    config.setdefault("training", {})
    if args.model_name:
        config["model"]["name_or_path"] = args.model_name
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps


def print_training_summary(config: dict[str, Any], args: argparse.Namespace) -> None:
    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    print("Whisper fine-tuning configuration")
    print(f"Model: {model.get('name_or_path')}")
    print(f"Train JSONL: {data.get('train_jsonl')}")
    print(f"Validation JSONL: {data.get('validation_jsonl')}")
    print(f"Test JSONL: {data.get('test_jsonl')}")
    print(f"Audio loading backend: {data.get('audio_loading_backend', 'librosa')}")
    print(f"Output dir: {training.get('output_dir')}")
    print(f"Batch size: {training.get('per_device_train_batch_size')}")
    print(f"Gradient accumulation: {training.get('gradient_accumulation_steps')}")
    print(f"FP16: {training.get('fp16')}")
    print(f"Gradient checkpointing: {training.get('gradient_checkpointing')}")
    print(f"Max steps: {training.get('max_steps')}")
    print(f"Training-time evaluation: {bool(training.get('run_eval_during_training', False))}")
    print(f"Run requested: {args.run}")


def validate_training_inputs(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    errors: list[str] = []
    reports = {}
    for key in ("train_jsonl", "validation_jsonl"):
        path = data.get(key)
        if not path:
            errors.append(f"missing_config_{key}")
            continue
        report = validate_whisper_jsonl(
            path,
            min_duration=float(data.get("min_duration_seconds", 0.3)),
            max_duration=float(data.get("max_duration_seconds", 30.0)),
        )
        reports[key] = report
        if not report["exists"]:
            errors.append(f"{key}_not_found:{path}")
        elif report["valid_rows"] == 0:
            errors.append(f"{key}_has_no_valid_rows:{path}")
    summary = {"reports": reports}
    if not errors:
        from readirect_asr.finetuning.whisper_dataset import load_whisper_dataset

        dataset = load_whisper_dataset(data["train_jsonl"], data["validation_jsonl"], data.get("test_jsonl"))
        summary = summarize_whisper_dataset(dataset)
    return {"errors": errors, "summary": summary}


def check_cuda() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else "CPU only",
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        return {"available": False, "gpu_name": "torch unavailable", "cuda_version": None, "error": str(exc)}


def run_training(config: dict[str, Any], args: argparse.Namespace) -> None:
    from datasets import DatasetDict, load_dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from readirect_asr.finetuning.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
    from readirect_asr.finetuning.whisper_audio import load_audio_array
    from readirect_asr.finetuning.whisper_generation_config import prepare_whisper_generation_config
    from readirect_asr.finetuning.whisper_metrics import build_compute_metrics

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    processor = WhisperProcessor.from_pretrained(
        model_cfg["name_or_path"],
        language=model_cfg.get("language", "English"),
        task=model_cfg.get("task", "transcribe"),
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["name_or_path"])
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    prepare_whisper_generation_config(
        model,
        processor,
        language="en" if str(model_cfg.get("language", "English")).lower().startswith("english") else model_cfg.get("language"),
        task=model_cfg.get("task", "transcribe"),
        suppress_tokens=model_cfg.get("suppress_tokens", []),
    )

    dataset = DatasetDict(
        {
            "train": load_dataset("json", data_files=str(data_cfg["train_jsonl"]), split="train"),
            "validation": load_dataset("json", data_files=str(data_cfg["validation_jsonl"]), split="train"),
        }
    )
    if data_cfg.get("test_jsonl") and Path(data_cfg["test_jsonl"]).exists():
        dataset["test"] = load_dataset("json", data_files=str(data_cfg["test_jsonl"]), split="train")
    if args.limit_train:
        dataset["train"] = dataset["train"].select(range(min(args.limit_train, len(dataset["train"]))))
    if args.limit_validation:
        dataset["validation"] = dataset["validation"].select(range(min(args.limit_validation, len(dataset["validation"]))))
    sampling_rate = int(data_cfg.get("sampling_rate", 16000))
    audio_column = data_cfg.get("audio_column", "audio")
    text_column = data_cfg.get("text_column", "sentence")
    audio_backend = data_cfg.get("audio_loading_backend", "librosa")

    def prepare_batch(batch):
        audio_array, sr = load_audio_array(batch[audio_column], sampling_rate=sampling_rate, backend=audio_backend)
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
        return batch

    vectorized = dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    run_eval = bool(train_cfg.get("run_eval_during_training", False))
    training_kwargs = build_seq2seq_training_arguments_kwargs(
        training_args_cls=Seq2SeqTrainingArguments,
        train_cfg=train_cfg,
        run_eval_during_training=run_eval,
    )
    args_train = Seq2SeqTrainingArguments(**training_kwargs)
    if not run_eval:
        print("Training-time evaluation is disabled. Run scripts/evaluate_finetuned_whisper.py after training.")
    trainer_kwargs = build_seq2seq_trainer_kwargs(
        trainer_cls=Seq2SeqTrainer,
        processor=processor,
        args=args_train,
        model=model,
        train_dataset=vectorized["train"],
        eval_dataset=vectorized["validation"] if run_eval else None,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id),
        compute_metrics=build_compute_metrics(processor) if run_eval else None,
    )
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    output_dir = Path(train_cfg["output_dir"])
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    metrics = train_result.metrics
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if run_eval:
        try:
            eval_metrics = trainer.evaluate(vectorized["validation"])
            (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"Warning: post-training evaluation failed, but model was saved. Run scripts/evaluate_finetuned_whisper.py separately. Error: {exc}")
    print(f"Training complete. Model saved to {output_dir}")


def build_seq2seq_training_arguments_kwargs(
    training_args_cls: Any,
    train_cfg: dict[str, Any],
    run_eval_during_training: bool,
) -> dict[str, Any]:
    parameters = inspect.signature(training_args_cls.__init__).parameters
    eval_strategy_value = train_cfg.get("evaluation_strategy", "steps" if run_eval_during_training else "no")
    kwargs = {
        "output_dir": train_cfg["output_dir"],
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 100)),
        "max_steps": int(train_cfg.get("max_steps", 1000)),
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "logging_steps": int(train_cfg.get("logging_steps", 25)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "predict_with_generate": bool(train_cfg.get("predict_with_generate", False) and run_eval_during_training),
        "generation_max_length": int(train_cfg.get("generation_max_length", 225)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "report_to": train_cfg.get("report_to", ["tensorboard"]),
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "seed": int(train_cfg.get("seed", 42)),
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = eval_strategy_value
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = eval_strategy_value
    return {key: value for key, value in kwargs.items() if key in parameters}


def build_seq2seq_trainer_kwargs(
    trainer_cls: Any,
    processor: Any,
    args: Any,
    model: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    compute_metrics: Any,
) -> dict[str, Any]:
    kwargs = {
        "args": args,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = getattr(processor, "tokenizer", processor)
    return kwargs


if __name__ == "__main__":
    raise SystemExit(main())
