from __future__ import annotations

from typing import Any


def generate_decision_report(
    readiness: dict[str, Any],
    metrics: dict[str, Any] | None,
    short_word_metrics: dict[str, Any] | None,
    decision: dict[str, Any],
    common_failures: list[dict[str, Any]] | None = None,
) -> str:
    metrics = metrics or {}
    short_word_metrics = short_word_metrics or {}
    common_failures = common_failures or []
    lines = [
        "# Fine-Tuning Decision Report",
        "",
        "## Executive Decision",
        "",
        f"- Decision: `{decision.get('decision')}`",
        f"- Confidence: `{decision.get('confidence')}`",
        f"- Suggested model: `{decision.get('suggested_model')}`",
        f"- Training mode: `{decision.get('suggested_training_mode')}`",
        "",
        "## Dataset Readiness",
        "",
        f"- Status: `{readiness.get('status')}`",
        f"- Ready: `{readiness.get('ready')}`",
        f"- Total rows: {readiness.get('total_rows')}",
        f"- Usable rows: {readiness.get('usable_rows')}",
        f"- Total hours: {readiness.get('total_hours')}",
        f"- Transcript coverage: {readiness.get('transcript_coverage')}",
        f"- Audio available rate: {readiness.get('audio_available_rate')}",
        f"- Issues: {readiness.get('issues', [])}",
        f"- Warnings: {readiness.get('warnings', [])}",
        "",
        "## Baseline ASR Performance",
        "",
        f"- WER: {metrics.get('wer')}",
        f"- CER: {metrics.get('cer')}",
        f"- Exact match rate: {metrics.get('exact_match_rate')}",
        f"- Evaluated rows: {metrics.get('evaluated_rows')}",
        f"- Blank hypotheses: {metrics.get('blank_hypothesis_count')}",
        "",
        "## ReaDirect Short-Word Performance",
        "",
        f"- Short-word rows: {short_word_metrics.get('total_short_word_rows')}",
        f"- Exact match rate: {short_word_metrics.get('exact_match_rate')}",
        f"- Average CER: {short_word_metrics.get('average_cer')}",
        f"- Blank rate: {short_word_metrics.get('blank_rate')}",
        f"- Near-match rate: {short_word_metrics.get('near_match_rate')}",
        "",
        "## Common Failure Patterns",
        "",
    ]
    if common_failures:
        lines.extend(f"- `{item.get('reference')}` -> `{item.get('hypothesis')}`: {item.get('count')}" for item in common_failures[:20])
    else:
        lines.append("- No common failure patterns available.")
    lines.extend(
        [
            "",
            "## Fine-Tuning Recommendation",
            "",
            *[f"- {reason}" for reason in decision.get("reasons", [])],
            "",
            "## Blocking Issues",
            "",
            *([f"- {issue}" for issue in decision.get("blocking_issues", [])] or ["- None."]),
            "",
            "## Suggested Next Steps",
            "",
            *[f"- {step}" for step in decision.get("recommended_next_steps", [])],
            "",
            "## Risks And Limitations",
            "",
            "- This report depends on the quality and coverage of the baseline ASR CSV.",
            "- Fine-tuning should not start until dataset licensing, privacy, and split quality are verified.",
            "- Short-word accuracy is weighted because early ReaDirect tasks rely on short Grade 1 words.",
        ]
    )
    return "\n".join(lines) + "\n"
