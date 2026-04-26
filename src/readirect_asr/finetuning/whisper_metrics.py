from __future__ import annotations

from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer, exact_match
from readirect_asr.text.normalization import normalize_for_wer


def normalize_prediction_text(text: str) -> str:
    return normalize_for_wer(text)


def compute_wer_metric(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return round(sum(compute_wer(ref, pred) for pred, ref in zip(predictions, references)) / len(references), 6)


def compute_cer_metric(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return round(sum(compute_cer(ref, pred) for pred, ref in zip(predictions, references)) / len(references), 6)


def compute_exact_match_rate(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return round(sum(1 for pred, ref in zip(predictions, references) if exact_match(ref, pred)) / len(references), 6)


def build_compute_metrics(processor: Any):
    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": compute_wer_metric(predictions, references),
            "cer": compute_cer_metric(predictions, references),
            "exact_match_rate": compute_exact_match_rate(predictions, references),
        }

    return compute_metrics
