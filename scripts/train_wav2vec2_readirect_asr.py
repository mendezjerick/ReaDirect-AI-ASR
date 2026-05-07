from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.text_normalization import normalize_asr_text
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ReaDirect Wav2Vec2 ASR model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--stage", choices=("mixed", "librispeech", "speechocean", "readirect_letters"), default="mixed")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--enable-light-eval", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def torch_runtime() -> dict[str, Any]:
    import torch

    cuda = bool(torch.cuda.is_available())
    return {
        "cuda": cuda,
        "device": "cuda" if cuda else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU",
        "cuda_version": torch.version.cuda,
    }


def artifact_stem(config: dict[str, Any]) -> str:
    output_name = Path(str(config["model"]["output_model_path"])).name
    if output_name == "wav2vec2-readirect-asr-letters-v2":
        return "wav2vec2_letters_v2"
    return "wav2vec2_readirect"


def dataset_mix(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        counts[dataset] = counts.get(dataset, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {dataset: round(count / total, 6) for dataset, count in sorted(counts.items())}


def final_training_loss(train_metrics: dict[str, Any]) -> float | None:
    for key in ("train_loss", "loss"):
        value = train_metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def write_training_log(config: dict[str, Any], lines: list[str]) -> None:
    output = resolve_repo_path(f"outputs/training/{artifact_stem(config)}_training_log.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_rows_for_training(rows: list[dict[str, Any]], config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    min_duration = float(data_cfg.get("min_duration_seconds", 0.2))
    max_duration = float(data_cfg.get("max_duration_seconds", 15.0))
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if stage != "mixed" and row.get("dataset") != stage:
            continue
        audio_path = str(row.get("audio_path", "")).strip()
        text = str(row.get("text", "")).strip()
        if not audio_path or not text:
            continue
        if not resolve_repo_path(audio_path).exists():
            continue
        try:
            duration = float(row.get("duration_seconds"))
        except (TypeError, ValueError):
            duration = None
        if duration is not None and duration < min_duration:
            continue
        if duration is not None and duration > max_duration:
            continue
        filtered.append(row)
    return filtered


def load_training_rows(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = config.get("data", {}).get("train_manifest")
    rows = filter_rows_for_training(read_jsonl(manifest), config, args.stage)
    if args.max_train_samples is not None:
        rows = rows[: args.max_train_samples]
    if args.smoke_test:
        limit = args.max_train_samples if args.max_train_samples is not None else 16
        rows = rows[: max(1, limit)]
    if not rows:
        raise RuntimeError(
            f"No usable training rows found for stage '{args.stage}' in {manifest}. "
            "Build and validate external_datasets/manifests/readirect_train_mixed.jsonl first."
        )
    return rows


def load_eval_rows(config: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    manifest = config.get("data", {}).get("valid_manifest")
    rows = filter_rows_for_training(read_jsonl(manifest), config, "mixed")
    return rows[:max_samples]


def load_audio_array(path: str | Path, sample_rate: int) -> tuple[Any, int]:
    import librosa

    audio, sr = librosa.load(str(resolve_repo_path(path)), sr=sample_rate, mono=True)
    return audio, int(sr)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(rows: list[dict[str, Any]], processor: Any, sample_rate: int):
    from datasets import Dataset

    vocab = set(processor.tokenizer.get_vocab().keys())
    dataset = Dataset.from_list(rows)

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio, sr = load_audio_array(batch["audio_path"], sample_rate)
        batch["input_values"] = processor(audio, sampling_rate=sr).input_values[0]
        batch["labels"] = processor.tokenizer(normalize_asr_text(batch["text"], vocab)).input_ids
        return batch

    return dataset.map(prepare_batch, remove_columns=dataset.column_names)


def resolve_fp16(value: Any, cuda_available: bool) -> bool:
    if isinstance(value, str) and value.lower() == "true_if_cuda_available":
        return cuda_available
    return bool(value) and cuda_available


def build_training_arguments(config: dict[str, Any], args: argparse.Namespace, cuda_available: bool, run_eval_during_training: bool):
    from transformers import TrainingArguments

    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation_during_training", {})
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    output_dir = resolve_repo_path(config["model"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = resolve_repo_path("outputs/training/logs")
    logging_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = str(eval_cfg.get("strategy", "steps" if run_eval_during_training else "no"))
    if not run_eval_during_training:
        eval_strategy = "no"

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(train_cfg.get("learning_rate", 3e-5)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 500)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "logging_steps": int(train_cfg.get("logging_steps", 50)),
        "eval_steps": int(eval_cfg.get("eval_steps", 1000)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "fp16": resolve_fp16(train_cfg.get("fp16", False), cuda_available),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "save_strategy": "steps",
        "report_to": ["tensorboard"],
        "seed": int(train_cfg.get("seed", 42)),
    }
    if args.smoke_test:
        kwargs.update({"max_steps": 2, "save_steps": 1, "logging_steps": 1, "warmup_steps": 0, "fp16": False})
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})


def build_trainer_kwargs(
    trainer_cls: Any,
    model: Any,
    processor: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    data_collator: Any,
) -> dict[str, Any]:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = processor
    return kwargs


def save_metadata(config: dict[str, Any], args: argparse.Namespace, rows: list[dict[str, Any]], train_metrics: dict[str, Any]) -> None:
    output_model_path = resolve_repo_path(config["model"]["output_model_path"])
    dataset_counts: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    output_name = Path(str(config["model"]["output_model_path"])).name
    configured_mix = config.get("data", {}).get("dataset_mix")
    if output_name == "wav2vec2-readirect-asr-letters-v2":
        metadata = {
            "model_name": "wav2vec2-readirect-asr-letters-v2",
            "base_model_path": config["model"]["base_model_path"],
            "output_model_path": config["model"]["output_model_path"],
            "training_type": "continued_fine_tuning",
            "training_mix": configured_mix
            or {
                "readirect_letters": 0.50,
                "speechocean": 0.30,
                "librispeech": 0.20,
            },
            "training_datasets": dataset_counts,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "evaluation_completed": False,
            "wer": None,
            "cer": None,
            "runtime_active": False,
            "training_metrics": train_metrics,
            "notes": [
                "This model is a continued fine-tuned v2 model focused on isolated English letter recognition.",
                "The v1 model was not overwritten.",
                "The phoneme model was not fine-tuned.",
                "Full evaluation must be run separately before making this model active.",
            ],
        }
        (output_model_path / "readirect_model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    metadata = {
        "base_model_path": config["model"]["base_model_path"],
        "training_datasets": dataset_counts,
        "dataset_weights": {
            "librispeech": config.get("data", {}).get("librispeech_weight"),
            "speechocean": config.get("data", {}).get("speechocean_weight"),
        },
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_config": config,
        "training_completed": True,
        "evaluation_completed": False,
        "final_wer": None,
        "final_cer": None,
        "validation_wer": None,
        "validation_cer": None,
        "training_metrics": train_metrics,
        "notes": "Training-first Wav2Vec2 ASR fine-tune. Evaluation is separate and metrics remain null until manually run.",
        "license_notes": "Verify LibriSpeech and SpeechOcean license terms before deployment.",
        "intended_use": "Letter-level and word-level ASR evidence for ReaDirect expected-centric scoring.",
        "limitations": "Fine-tuning does not replace phoneme evidence, expected-centric scoring, or letter-confusion logic.",
    }
    (output_model_path / "readirect_model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def save_training_summary(
    config: dict[str, Any],
    args: argparse.Namespace,
    runtime: dict[str, Any],
    train_row_count: int,
    train_metrics: dict[str, Any],
    light_eval_metrics: dict[str, Any] | None,
    start_time: datetime,
    end_time: datetime,
    rows: list[dict[str, Any]],
) -> None:
    output = resolve_repo_path(f"outputs/training/{artifact_stem(config)}_training_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    train_cfg = config.get("training", {})
    summary = {
        "training_completed": True,
        "config": str(args.config),
        "stage": args.stage,
        "smoke_test": bool(args.smoke_test),
        "train_rows": train_row_count,
        "base_model_path": config["model"]["base_model_path"],
        "output_model_path": config["model"]["output_model_path"],
        "checkpoint_dir": config["model"]["checkpoint_dir"],
        "train_manifest": config.get("data", {}).get("train_manifest"),
        "number_of_train_rows": train_row_count,
        "dataset_mix_target": config.get("data", {}).get("dataset_mix"),
        "dataset_mix_actual": dataset_mix(rows),
        "num_train_epochs": train_cfg.get("num_train_epochs"),
        "learning_rate": train_cfg.get("learning_rate"),
        "batch_size": train_cfg.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
        "fp16_used": False if args.smoke_test else resolve_fp16(train_cfg.get("fp16", False), bool(runtime.get("cuda"))),
        "cuda_used": bool(runtime.get("cuda")),
        "gpu_name": runtime.get("gpu_name"),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_training_time": str(end_time - start_time),
        "final_training_loss": final_training_loss(train_metrics),
        "evaluation_completed": False,
        "wer": None,
        "cer": None,
        "note": "Evaluation is separate and was not run during training.",
        "runtime": runtime,
        "evaluation_during_training_enabled": bool(config.get("evaluation_during_training", {}).get("enabled", False)) and not args.no_eval,
        "light_eval_metrics": light_eval_metrics,
        "train_metrics": train_metrics,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def maybe_run_light_eval(trainer: Any, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    try:
        return dict(trainer.evaluate())
    except Exception as exc:
        print(f"Warning: light evaluation failed after model save. Training output remains saved. Error: {exc}")
        return {"error": str(exc)}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    start_time = datetime.now(timezone.utc)
    log_lines = [
        f"start_time={start_time.isoformat()}",
        f"config={args.config}",
        f"smoke_test={args.smoke_test}",
        f"no_eval={args.no_eval}",
    ]
    runtime = torch_runtime()
    print(f"torch.cuda.is_available(): {runtime['cuda']}")
    print(f"Runtime device: {runtime['device']} ({runtime['gpu_name']})")
    log_lines.append(f"torch.cuda.is_available={runtime['cuda']}")
    log_lines.append(f"gpu_name={runtime['gpu_name']}")
    require_cuda = bool(args.require_cuda) or bool(config.get("training", {}).get("require_cuda", False))
    if require_cuda and not runtime["cuda"]:
        print("CUDA is required but unavailable. Stopping before training.")
        log_lines.append("stopped=CUDA is required but unavailable")
        write_training_log(config, log_lines)
        return 2
    if not runtime["cuda"]:
        print("Warning: CUDA is unavailable. CPU training is supported for smoke tests but full training will be slow.")

    from transformers import Trainer, Wav2Vec2ForCTC, Wav2Vec2Processor

    model_cfg = config["model"]
    data_cfg = config["data"]
    base_model_path = resolve_repo_path(model_cfg["base_model_path"])
    output_model_path = resolve_repo_path(model_cfg["output_model_path"])
    if base_model_path.resolve() == output_model_path.resolve():
        raise RuntimeError("Output model path must be separate from the base model path. Refusing to overwrite the base model.")
    output_model_path.mkdir(parents=True, exist_ok=True)

    train_rows = load_training_rows(config, args)
    log_lines.append(f"train_rows={len(train_rows)}")
    log_lines.append(f"dataset_mix_actual={dataset_mix(train_rows)}")
    eval_cfg = config.get("evaluation_during_training", {})
    run_eval_during_training = bool(eval_cfg.get("enabled", False)) and not args.no_eval
    light_eval_enabled = bool(args.enable_light_eval) and not args.no_eval

    processor = Wav2Vec2Processor.from_pretrained(str(base_model_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(base_model_path), local_files_only=True)
    if bool(config.get("training", {}).get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()
    if bool(config.get("training", {}).get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    sample_rate = int(data_cfg.get("sample_rate", 16000))
    train_dataset = prepare_dataset(train_rows, processor, sample_rate)
    eval_dataset = None
    if run_eval_during_training or light_eval_enabled:
        max_eval_samples = int(eval_cfg.get("max_eval_samples", 100))
        eval_rows = load_eval_rows(config, max_eval_samples)
        if eval_rows:
            eval_dataset = prepare_dataset(eval_rows, processor, sample_rate)
        else:
            print("Warning: light validation was requested but no valid rows were available. Continuing train-only.")
            run_eval_during_training = False
            light_eval_enabled = False

    training_args = build_training_arguments(config, args, runtime["cuda"], run_eval_during_training)
    trainer = Trainer(
        **build_trainer_kwargs(
            Trainer,
            model,
            processor,
            training_args,
            train_dataset,
            eval_dataset if run_eval_during_training else None,
            DataCollatorCTCWithPadding(processor=processor),
        )
    )
    if not run_eval_during_training:
        print("Training-time evaluation is disabled. Full evaluation must be run with scripts/evaluate_wav2vec2_readirect_asr.py.")
        log_lines.append("training_time_evaluation=disabled")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = dict(train_result.metrics)
    log_lines.append(f"train_metrics={json.dumps(train_metrics, sort_keys=True)}")
    trainer.save_model(str(output_model_path))
    processor.save_pretrained(str(output_model_path))
    log_lines.append(f"saved_model={output_model_path.relative_to(PROJECT_ROOT)}")
    light_eval_metrics = None
    if light_eval_enabled and eval_dataset is not None:
        trainer.eval_dataset = eval_dataset
        light_eval_metrics = maybe_run_light_eval(trainer, True)
        log_lines.append(f"light_eval_metrics={json.dumps(light_eval_metrics, sort_keys=True)}")

    save_metadata(config, args, train_rows, train_metrics)
    end_time = datetime.now(timezone.utc)
    save_training_summary(config, args, runtime, len(train_rows), train_metrics, light_eval_metrics, start_time, end_time, train_rows)
    log_lines.append(f"end_time={end_time.isoformat()}")
    log_lines.append("evaluation_completed=false")
    log_lines.append("wer=null")
    log_lines.append("cer=null")
    write_training_log(config, log_lines)
    print(f"Training complete. Model and processor saved to {output_model_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
