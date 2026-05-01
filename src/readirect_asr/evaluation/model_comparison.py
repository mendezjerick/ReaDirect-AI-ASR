from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer


NOISE_PATTERN = re.compile(r"\[[^\]]+\]|\([^\)]+\)|<[^>]+>")
WORD_PATTERN = re.compile(r"[^a-z0-9'\s]")

LETTER_ALIASES: dict[str, set[str]] = {
    "a": {"a", "ay", "aye"},
    "b": {"b", "be", "bee"},
    "c": {"c", "see", "sea"},
    "d": {"d", "dee"},
    "e": {"e", "ee"},
    "f": {"f", "ef", "eff"},
    "g": {"g", "gee", "jee"},
    "h": {"h", "aitch", "haitch"},
    "i": {"i", "eye"},
    "j": {"j", "jay"},
    "k": {"k", "kay"},
    "l": {"l", "el", "ell", "elle"},
    "m": {"m", "em", "emm"},
    "n": {"n", "en", "enn"},
    "o": {"o", "oh", "owe"},
    "p": {"p", "pee"},
    "q": {"q", "cue", "queue", "kyu", "kyoo"},
    "r": {"r", "are", "ar"},
    "s": {"s", "ess", "es"},
    "t": {"t", "tee", "tea"},
    "u": {"u", "you", "yu", "yoo"},
    "v": {"v", "vee"},
    "w": {"w", "double you", "double u", "doubleu", "dubya"},
    "x": {"x", "ex", "axe", "eks"},
    "y": {"y", "why"},
    "z": {"z", "zee", "zed", "zi", "zii", "zih", "zy", "zey", "ze"},
}

DANGEROUS_TRANSCRIPT_ONLY_LETTER_REJECTS: dict[str, set[str]] = {
    "q": {"you", "u", "yoo", "yu"},
}

EXPECTED_WORD_CONFUSIONS: dict[str, set[str]] = {
    "ten": {"then"},
    "red": {"read"},
    "read": {"red"},
    "tree": {"three"},
    "three": {"tree"},
    "see": {"sea", "c"},
    "sea": {"see", "c"},
    "right": {"write"},
    "write": {"right"},
    "hear": {"here"},
    "here": {"hear"},
    "be": {"bee", "b"},
    "bee": {"be", "b"},
}


@dataclass(frozen=True)
class CorrectionResult:
    normalized_transcript: str
    corrected_transcript: str
    displayed_transcript: str
    corrected_wer: float
    corrected_cer: float
    corrected_exact_match: bool
    accepted: bool
    reason: str


def normalize_eval_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    value = NOISE_PATTERN.sub(" ", value)
    value = value.lower()
    value = WORD_PATTERN.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def infer_prompt_type(expected_text: str) -> str:
    normalized = normalize_eval_text(expected_text)
    if len(normalized) == 1 and normalized.isalpha():
        return "letter"
    if normalized and len(normalized.split()) == 1:
        return "word"
    return "sentence"


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_eval_text(reference) == normalize_eval_text(hypothesis)


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    return {
        "wer": compute_wer(normalize_eval_text(reference), normalize_eval_text(hypothesis)),
        "cer": compute_cer(normalize_eval_text(reference), normalize_eval_text(hypothesis)),
        "exact_match": exact_match(reference, hypothesis),
    }


def correct_expected_centric(
    expected_text: str,
    raw_transcript: str,
    prompt_type: str | None = None,
    *,
    use_correction: bool = True,
    use_phoneme_evidence: bool = False,
) -> CorrectionResult:
    normalized_expected = normalize_eval_text(expected_text)
    normalized_raw = normalize_eval_text(raw_transcript)
    active_prompt_type = prompt_type or infer_prompt_type(normalized_expected)
    raw_score = score_pair(normalized_expected, normalized_raw)

    if not use_correction:
        return CorrectionResult(
            normalized_transcript=normalized_raw,
            corrected_transcript=normalized_raw,
            displayed_transcript=normalized_raw,
            corrected_wer=raw_score["wer"],
            corrected_cer=raw_score["cer"],
            corrected_exact_match=raw_score["exact_match"],
            accepted=raw_score["exact_match"],
            reason="correction_disabled",
        )

    if not normalized_expected:
        return _raw_result(normalized_raw, raw_score, "empty_expected_text")
    if not normalized_raw:
        return _raw_result(normalized_raw, raw_score, "empty_raw_transcript")
    if normalized_expected == normalized_raw:
        return _accepted(normalized_expected, normalized_raw, "exact_match_after_normalization")

    if active_prompt_type == "letter":
        expected_letter = normalized_expected
        compact_raw = normalized_raw.replace(" ", "").replace("-", "")
        dangerous = compact_raw in DANGEROUS_TRANSCRIPT_ONLY_LETTER_REJECTS.get(expected_letter, set())
        if dangerous and not use_phoneme_evidence:
            return _raw_result(normalized_raw, raw_score, "transcript_only_rejected_ambiguous_letter_alias")
        aliases = {alias.replace(" ", "").replace("-", "") for alias in LETTER_ALIASES.get(expected_letter, set())}
        if compact_raw in aliases:
            return _accepted(normalized_expected, normalized_raw, "expected_centric_letter_alias_match")
        return _raw_result(normalized_raw, raw_score, "no_letter_alias_match")

    if active_prompt_type == "word":
        if len(normalized_expected.split()) == 1 and len(normalized_raw.split()) == 1:
            if normalized_raw in EXPECTED_WORD_CONFUSIONS.get(normalized_expected, set()):
                return _accepted(normalized_expected, normalized_raw, "expected_centric_known_word_confusion")
        return _raw_result(normalized_raw, raw_score, "no_safe_word_correction")

    return _raw_result(normalized_raw, raw_score, "sentence_correction_not_applied")


def _accepted(expected: str, normalized_raw: str, reason: str) -> CorrectionResult:
    return CorrectionResult(
        normalized_transcript=normalized_raw,
        corrected_transcript=expected,
        displayed_transcript=expected,
        corrected_wer=0.0,
        corrected_cer=0.0,
        corrected_exact_match=True,
        accepted=True,
        reason=reason,
    )


def _raw_result(normalized_raw: str, raw_score: dict[str, Any], reason: str) -> CorrectionResult:
    return CorrectionResult(
        normalized_transcript=normalized_raw,
        corrected_transcript=normalized_raw,
        displayed_transcript=normalized_raw,
        corrected_wer=float(raw_score["wer"]),
        corrected_cer=float(raw_score["cer"]),
        corrected_exact_match=bool(raw_score["exact_match"]),
        accepted=bool(raw_score["exact_match"]),
        reason=reason,
    )


def winner_for(left: float, right: float, lower_is_better: bool = True) -> str:
    if abs(left - right) < 1e-12:
        return "tie"
    if lower_is_better:
        return "wav2vec2" if left < right else "whisper"
    return "wav2vec2" if left > right else "whisper"


def aggregate_rows(rows: list[dict[str, Any]], group_key: str | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows} if group_key is None else defaultdict(list)
    if group_key is not None:
        for row in rows:
            groups[str(row.get(group_key, "unknown") or "unknown")].append(row)

    return {name: _aggregate_group(group_rows) for name, group_rows in groups.items()}


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "wav2vec2": _aggregate_model(rows, "wav2vec2"),
        "whisper": _aggregate_model(rows, "whisper"),
        "winners": _winner_summary(rows),
    }


def _aggregate_model(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ok_rows = [row for row in rows if not row.get(f"{prefix}_error")]
    references = [str(row.get("normalized_expected", "")) for row in ok_rows]
    raw_predictions = [str(row.get(f"{prefix}_normalized_transcript", "")) for row in ok_rows]
    corrected_predictions = [str(row.get(f"{prefix}_corrected_transcript", "")) for row in ok_rows]
    return {
        "rows": len(ok_rows),
        "raw_wer": _corpus_wer(references, raw_predictions),
        "corrected_wer": _corpus_wer(references, corrected_predictions),
        "raw_cer": _corpus_cer(references, raw_predictions),
        "corrected_cer": _corpus_cer(references, corrected_predictions),
        "raw_exact_match_rate": _rate([bool(row[f"{prefix}_exact_match"]) for row in ok_rows]),
        "corrected_exact_match_rate": _rate([bool(row[f"{prefix}_corrected_exact_match"]) for row in ok_rows]),
        "accepted_rate": _rate([bool(row[f"{prefix}_accepted"]) for row in ok_rows]),
        "error_rows": len(rows) - len(ok_rows),
    }


def _winner_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "raw": dict(Counter(str(row.get("winner_raw", "unknown")) for row in rows)),
        "corrected": dict(Counter(str(row.get("winner_corrected", "unknown")) for row in rows)),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(sum(1 for value in values if value) / len(values), 6) if values else 0.0


def _corpus_wer(references: list[str], predictions: list[str]) -> float:
    if not references:
        return 0.0
    try:
        from jiwer import wer

        return round(float(wer(references, predictions)), 6)
    except Exception:
        joined_ref = " ".join(references)
        joined_pred = " ".join(predictions)
        return round(compute_wer(joined_ref, joined_pred), 6)


def _corpus_cer(references: list[str], predictions: list[str]) -> float:
    if not references:
        return 0.0
    try:
        from jiwer import cer

        return round(float(cer(references, predictions)), 6)
    except Exception:
        joined_ref = " ".join(references)
        joined_pred = " ".join(predictions)
        return round(compute_cer(joined_ref, joined_pred), 6)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fieldnames:
            file.write("")
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Wav2Vec2 vs Whisper Comparison",
        "",
        f"- Manifest: `{summary.get('manifest')}`",
        f"- Rows requested: {summary.get('rows_requested')}",
        f"- Rows evaluated: {summary.get('rows_evaluated')}",
        f"- Fair comparison: {summary.get('fair_comparison')}",
        f"- Correction enabled: {summary.get('use_correction')}",
        f"- Phoneme evidence: {summary.get('phoneme_evidence')}",
        "",
        "## Overall",
        "",
        _comparison_table(summary["overall"]["overall"]),
        "",
        "## By Prompt Type",
        "",
    ]
    for group, data in sorted(summary.get("by_prompt_type", {}).items()):
        lines.extend([f"### {group}", "", _comparison_table(data), ""])
    lines.extend(["## By Dataset", ""])
    for group, data in sorted(summary.get("by_dataset", {}).items()):
        lines.extend([f"### {group}", "", _comparison_table(data), ""])
    lines.extend(["## Winner Table", "", _winner_table(summary), ""])
    lines.extend(["## Recommendation", "", summary.get("recommendation", "Run the comparison to generate a recommendation."), ""])
    lines.extend(["## Notes", ""])
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_table(group: dict[str, Any]) -> str:
    lines = [
        "| Model | Rows | Raw WER | Corrected WER | Raw CER | Corrected CER | Raw Exact | Corrected Exact | Accepted | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("wav2vec2", "whisper"):
        data = group[model]
        lines.append(
            f"| {model} | {data['rows']} | {data['raw_wer']:.6f} | {data['corrected_wer']:.6f} | "
            f"{data['raw_cer']:.6f} | {data['corrected_cer']:.6f} | {data['raw_exact_match_rate']:.6f} | "
            f"{data['corrected_exact_match_rate']:.6f} | {data['accepted_rate']:.6f} | {data['error_rows']} |"
        )
    return "\n".join(lines)


def _winner_table(summary: dict[str, Any]) -> str:
    rows = []
    overall = summary["overall"]["overall"]
    rows.append(("overall raw WER", winner_for(overall["wav2vec2"]["raw_wer"], overall["whisper"]["raw_wer"])))
    rows.append(("overall corrected WER", winner_for(overall["wav2vec2"]["corrected_wer"], overall["whisper"]["corrected_wer"])))
    rows.append(("overall exact match", winner_for(overall["wav2vec2"]["raw_exact_match_rate"], overall["whisper"]["raw_exact_match_rate"], lower_is_better=False)))
    rows.append(("overall corrected exact match", winner_for(overall["wav2vec2"]["corrected_exact_match_rate"], overall["whisper"]["corrected_exact_match_rate"], lower_is_better=False)))
    for prompt_type, data in sorted(summary.get("by_prompt_type", {}).items()):
        rows.append((prompt_type, winner_for(data["wav2vec2"]["corrected_wer"], data["whisper"]["corrected_wer"])))
    lines = ["| Category | Winner |", "|---|---|"]
    lines.extend(f"| {category} | {winner} |" for category, winner in rows)
    return "\n".join(lines)


def recommendation_from_summary(summary: dict[str, Any]) -> str:
    by_type = summary.get("by_prompt_type", {})
    if not by_type:
        return "No prompt-type groups were available, so no routing recommendation is made."
    winners = {
        key: winner_for(value["wav2vec2"]["corrected_wer"], value["whisper"]["corrected_wer"])
        for key, value in by_type.items()
    }
    letter = winners.get("letter")
    word = winners.get("word")
    sentence = winners.get("sentence")
    if letter == "wav2vec2" and word == "wav2vec2" and sentence == "whisper":
        return "Recommended routing: letters -> Wav2Vec2, words -> Wav2Vec2, sentences -> Whisper."
    if letter == "wav2vec2" and word == "wav2vec2" and sentence == "wav2vec2":
        return "Recommended routing: Wav2Vec2 can be primary for all evaluated categories; keep Whisper as fallback for sentence robustness/debugging."
    if sentence == "whisper":
        return "Recommended routing: keep Whisper for sentences and use Wav2Vec2 where it wins short-form prompts."
    return "Recommended routing: results are mixed; use hybrid routing by prompt type and keep fallback behavior."
