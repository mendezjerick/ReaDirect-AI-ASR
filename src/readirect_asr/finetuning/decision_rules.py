from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "wer_recommend_threshold": 0.20,
    "cer_recommend_threshold": 0.10,
    "wer_good_threshold": 0.10,
    "cer_good_threshold": 0.05,
    "short_word_accuracy_threshold": 0.75,
    "short_word_good_threshold": 0.85,
    "blank_hypothesis_rate_threshold": 0.05,
    "suggested_model": "openai/whisper-base.en",
}


def decide_finetuning_need(
    metrics: dict[str, Any] | None,
    readiness: dict[str, Any],
    short_word_metrics: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []
    blocking_issues: list[str] = []
    next_steps: list[str] = []
    if not metrics:
        return {
            "decision": "baseline_missing",
            "confidence": "high",
            "reasons": ["No baseline ASR metrics were available."],
            "blocking_issues": ["baseline_missing"],
            "recommended_next_steps": ["Run the faster-whisper baseline before deciding whether to fine-tune."],
            "suggested_model": cfg["suggested_model"],
            "suggested_training_mode": "local_gpu_optional",
        }
    if not readiness.get("ready", False):
        return {
            "decision": "more_data_needed",
            "confidence": "high",
            "reasons": ["Dataset readiness checks did not pass."],
            "blocking_issues": list(readiness.get("issues", [])),
            "recommended_next_steps": list(readiness.get("recommendations", [])) or ["Improve dataset readiness before training."],
            "suggested_model": cfg["suggested_model"],
            "suggested_training_mode": "local_gpu_optional",
        }

    wer = float(metrics.get("wer", 0.0) or 0.0)
    cer = float(metrics.get("cer", 0.0) or 0.0)
    exact = float(metrics.get("exact_match_rate", 0.0) or 0.0)
    total = max(float(metrics.get("evaluated_rows", metrics.get("total_rows", 0)) or 0), 1.0)
    blank_rate = float(metrics.get("blank_hypothesis_count", 0.0) or 0.0) / total
    short = short_word_metrics or {}
    short_exact = float(short.get("exact_match_rate", 1.0) or 0.0)

    recommend = False
    confidence = "medium"
    if wer > cfg["wer_recommend_threshold"] and cer > cfg["cer_recommend_threshold"]:
        recommend = True
        confidence = "high"
        reasons.append("WER and CER are above fine-tuning recommendation thresholds.")
    if short and short_exact < cfg["short_word_accuracy_threshold"]:
        recommend = True
        reasons.append("Short-word exact match is below the ReaDirect threshold.")
    if blank_rate > cfg["blank_hypothesis_rate_threshold"]:
        recommend = True
        reasons.append("Blank ASR hypothesis rate is high.")

    if (
        wer <= cfg["wer_good_threshold"]
        and cer <= cfg["cer_good_threshold"]
        and short_exact >= cfg["short_word_good_threshold"]
        and blank_rate <= cfg["blank_hypothesis_rate_threshold"]
    ):
        return {
            "decision": "not_needed_yet",
            "confidence": "high",
            "reasons": ["Baseline ASR performance is already within good thresholds."],
            "blocking_issues": [],
            "recommended_next_steps": ["Continue collecting labeled data and monitor ReaDirect short-word accuracy."],
            "suggested_model": cfg["suggested_model"],
            "suggested_training_mode": "local_gpu_optional",
        }

    if recommend:
        next_steps.append("Prepare Whisper-compatible JSONL splits and run a small guarded training experiment later.")
        next_steps.append("Review common short-word confusions before training.")
        return {
            "decision": "fine_tuning_recommended",
            "confidence": confidence,
            "reasons": reasons or ["Baseline metrics are mixed and likely need domain adaptation."],
            "blocking_issues": blocking_issues,
            "recommended_next_steps": next_steps,
            "suggested_model": cfg["suggested_model"],
            "suggested_training_mode": "local_gpu_optional",
        }

    return {
        "decision": "not_needed_yet" if exact >= 0.75 else "fine_tuning_recommended",
        "confidence": "medium",
        "reasons": ["Metrics are mixed; continue evaluation before committing to training."],
        "blocking_issues": [],
        "recommended_next_steps": ["Run a larger baseline sample and inspect ReaDirect short-word errors."],
        "suggested_model": cfg["suggested_model"],
        "suggested_training_mode": "local_gpu_optional",
    }
