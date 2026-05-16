from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer
from readirect_asr.scoring.phoneme_comparison import phoneme_similarity
from readirect_asr.text.normalization import normalize_for_wer
from readirect_asr.text.transcript_normalizer import (
    KNOWN_ASR_CONFUSIONS,
    LETTER_PRONUNCIATIONS,
    detect_prompt_type,
    generate_expected_phonemes,
)


DEFAULT_DYNAMIC_CONFIG = {
    "enabled": True,
    "letter_accept_threshold": 0.72,
    "word_accept_threshold": 0.78,
    "rhyme_accept_threshold": 0.78,
    "sentence_word_accept_threshold": 0.80,
    "passage_word_accept_threshold": 0.82,
    "homophone_threshold": 0.96,
    "min_phoneme_for_low_text_match": 0.90,
    "min_gop_for_acceptance": 0.75,
    "skip_on_retry_required": True,
    "skip_on_uncertain_audio": True,
    "debug": False,
}

SHORT_PROMPT_TYPES = {"letter", "word", "rhyme", "rhyming_word"}
LONG_PROMPT_TYPES = {"sentence", "paragraph", "passage", "reading_passage", "final_sentence", "final_passage"}
FUNCTION_WORDS = {"a", "an", "the", "to", "of", "in", "on", "is", "it", "he", "she", "we"}


@dataclass(frozen=True)
class DynamicCorrectionResult:
    accepted: bool
    corrected_text: str | None
    display_text: str
    strategy: str
    sub_strategy: str
    confidence: float
    threshold: float
    spelling_similarity: float
    phoneme_similarity: float | None
    gop_score: float | None
    homophone_match: bool
    context_score: float
    known_confusion_score: float
    reason: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "corrected_text": self.corrected_text,
            "display_text": self.display_text,
            "strategy": self.strategy,
            "sub_strategy": self.sub_strategy,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "spelling_similarity": self.spelling_similarity,
            "phoneme_similarity": self.phoneme_similarity,
            "gop_score": self.gop_score,
            "homophone_match": self.homophone_match,
            "context_score": self.context_score,
            "known_confusion_score": self.known_confusion_score,
            "reason": self.reason,
            "enabled": self.enabled,
        }


def correct_expected_word(
    expected_text: str,
    raw_transcript: str,
    prompt_type: str,
    gop_score: float | None = None,
    phoneme_similarity_score: float | None = None,
    audio_quality: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
    context_metadata: dict[str, Any] | None = None,
    debug: bool = False,
    *,
    cmudict_loader: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_config = _config(config)
    detected_prompt_type = detect_prompt_type(expected_text, prompt_type=prompt_type)
    threshold = _threshold_for_prompt(detected_prompt_type, active_config)
    expected = str(expected_text or "").strip()
    raw = str(raw_transcript or "").strip()
    normalized_expected = normalize_for_wer(expected)
    normalized_raw = normalize_for_wer(raw)
    context = context_metadata or {}
    context_score = _context_score(context)

    if not bool(active_config["enabled"]):
        return _result(False, None, raw, detected_prompt_type, "disabled", 0.0, threshold, 0.0, None, gop_score, False, context_score, 0.0, "dynamic expected-word correction is disabled", enabled=False)
    if bool(retry_required) and bool(active_config["skip_on_retry_required"]):
        return _result(False, None, raw, detected_prompt_type, "skipped_retry_required", 0.0, threshold, 0.0, None, gop_score, False, context_score, 0.0, "retry_required is true")
    if bool(uncertain) and bool(active_config["skip_on_uncertain_audio"]):
        return _result(False, None, raw, detected_prompt_type, "skipped_uncertain_audio", 0.0, threshold, 0.0, None, gop_score, False, context_score, 0.0, "uncertain audio is true")
    if not normalized_expected:
        return _result(False, None, raw, detected_prompt_type, "skipped_no_expected_text", 0.0, threshold, 0.0, None, gop_score, False, context_score, 0.0, "expected_text is missing")
    if not normalized_raw:
        return _result(False, None, raw, detected_prompt_type, "rejected_blank_raw_transcript", 0.0, threshold, 0.0, None, gop_score, False, context_score, 0.0, "raw transcript is blank")

    spelling = _spelling_similarity(normalized_expected, normalized_raw)
    phoneme_score = phoneme_similarity_score
    if phoneme_score is None:
        phoneme_score = _phoneme_similarity(normalized_expected, normalized_raw, cmudict_loader)
    known_confusion = _known_confusion_score(normalized_expected, normalized_raw, detected_prompt_type)
    homophone = bool(phoneme_score is not None and phoneme_score >= float(active_config["homophone_threshold"]) and normalized_expected != normalized_raw)

    if _exact_match(normalized_expected, normalized_raw):
        return _result(True, expected, expected, detected_prompt_type, "exact_normalized_match", 1.0, threshold, 1.0, phoneme_score, gop_score, False, context_score, known_confusion, "raw transcript matches expected text after normalization")

    if detected_prompt_type == "letter" and _letter_alias_match(normalized_expected, normalized_raw):
        return _result(True, expected, expected, detected_prompt_type, "letter_alias_match", 0.98, threshold, spelling, phoneme_score, gop_score, False, context_score, max(known_confusion, 0.9), "raw transcript is a spoken form of the expected letter")

    final_score = _weighted_score(
        prompt_type=detected_prompt_type,
        spelling_similarity=spelling,
        phoneme_similarity_score=phoneme_score,
        gop_score=gop_score,
        context_score=context_score,
        known_confusion_score=known_confusion,
    )
    accepted, sub_strategy, reason = _decision(
        expected=normalized_expected,
        raw=normalized_raw,
        prompt_type=detected_prompt_type,
        final_score=final_score,
        threshold=threshold,
        spelling=spelling,
        phoneme_score=phoneme_score,
        gop_score=gop_score,
        context_score=context_score,
        known_confusion=known_confusion,
        homophone=homophone,
        config=active_config,
    )
    if accepted:
        if sub_strategy == "homophone_match" and phoneme_score is not None:
            final_score = max(final_score, phoneme_score)
        elif sub_strategy == "gop_supported_expected_match" and gop_score is not None:
            final_score = max(final_score, gop_score)
        elif sub_strategy == "spelling_context_expected_match":
            final_score = max(final_score, spelling)
        elif sub_strategy == "known_asr_confusion_match":
            final_score = max(final_score, known_confusion)

    return _result(
        accepted=accepted,
        corrected=expected if accepted else None,
        display=expected if accepted else raw,
        prompt_type=detected_prompt_type,
        sub_strategy=sub_strategy,
        confidence=final_score,
        threshold=threshold,
        spelling=spelling,
        phoneme=phoneme_score,
        gop=gop_score,
        homophone=homophone,
        context=context_score,
        known=known_confusion,
        reason=reason,
    )


def apply_dynamic_expected_word_correction(
    transcript_meta: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    audio_quality: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
    context_metadata: dict[str, Any] | None = None,
    cmudict_loader: Any | None = None,
) -> dict[str, Any]:
    updated = dict(transcript_meta)
    expected = str(updated.get("expected_text") or "")
    raw = str(updated.get("raw_transcript") or updated.get("wav2vec2_transcript") or "")
    prompt_type = str(updated.get("prompt_type") or "unknown")
    gop_score = _optional_float(updated.get("gop_score"))
    existing_phoneme = _optional_float(updated.get("phonetic_similarity_score"))
    context = context_metadata or {}
    detected_prompt_type = detect_prompt_type(expected, prompt_type=prompt_type)

    if _critical_phoneme_contradicted(updated):
        result = _result(False, None, raw, detected_prompt_type, "skipped_critical_phoneme_contradiction", 0.0, _threshold_for_prompt(detected_prompt_type, _config(config)), 0.0, existing_phoneme, gop_score, False, _context_score(context), 0.0, "critical phoneme evidence contradicted expected answer")
        return _with_dynamic_fields(updated, DynamicCorrectionResult(**result))

    if detected_prompt_type in LONG_PROMPT_TYPES:
        alignment = dynamic_word_alignment(
            expected,
            raw,
            detected_prompt_type,
            gop_word_scores=list(updated.get("gop_word_scores", []) or []),
            config=config,
            retry_required=retry_required,
            uncertain=uncertain,
            cmudict_loader=cmudict_loader,
        )
        applied_items = [item for item in alignment if item.get("status") in {"accepted_by_dynamic_expected_word_correction", "accepted_by_homophone"}]
        best = max((float(item.get("dynamic_correction_confidence", 0.0) or 0.0) for item in applied_items), default=0.0)
        result = _result(
            accepted=bool(applied_items),
            corrected=None,
            display=raw,
            prompt_type=detected_prompt_type,
            sub_strategy="word_alignment_dynamic_correction" if applied_items else "no_word_alignment_corrections",
            confidence=best,
            threshold=_threshold_for_prompt(detected_prompt_type, _config(config)),
            spelling=0.0,
            phoneme=None,
            gop=gop_score,
            homophone=any(item.get("status") == "accepted_by_homophone" for item in applied_items),
            context=_context_score(context),
            known=0.0,
            reason="dynamic correction applied to aligned words only" if applied_items else "no aligned word met dynamic correction threshold",
        )
        updated = _with_dynamic_fields(updated, DynamicCorrectionResult(**result))
        updated["word_alignment"] = alignment
        debug_metadata = dict(updated.get("debug_metadata", {}) or {})
        debug_metadata["word_alignment"] = alignment
        debug_metadata["dynamic_expected_word_correction"] = result
        updated["debug_metadata"] = debug_metadata
        return updated

    result = correct_expected_word(
        expected_text=expected,
        raw_transcript=raw,
        prompt_type=detected_prompt_type,
        gop_score=gop_score,
        phoneme_similarity_score=existing_phoneme if existing_phoneme and existing_phoneme > 0 else None,
        audio_quality=audio_quality,
        retry_required=retry_required,
        uncertain=uncertain,
        context_metadata=context,
        cmudict_loader=cmudict_loader,
        config=config,
    )
    updated = _with_dynamic_fields(updated, DynamicCorrectionResult(**result))
    if result["accepted"] and not bool(updated.get("accepted", False)):
        updated["corrected_transcript"] = expected
        updated["displayed_transcript"] = expected
        updated["accepted"] = True
        updated["normalization_applied"] = True
        updated["normalization_reason"] = result["reason"]
        updated["correction_strategy_used"] = "dynamic_expected_word_gop_match" if result.get("gop_score") is not None and float(result.get("gop_score") or 0.0) >= float(_config(config)["min_gop_for_acceptance"]) else "dynamic_expected_word_correction"
        updated["accepted_by_phoneme_evidence"] = bool((result.get("phoneme_similarity") or 0.0) >= result["threshold"] or result.get("homophone_match"))
        updated["corrected_wer"] = compute_wer(expected, expected)
        updated["corrected_cer"] = compute_cer(expected, expected)
        updated["composite_score"] = max(float(updated.get("composite_score", 0.0) or 0.0), float(result["confidence"]))
    return updated


def dynamic_word_alignment(
    expected_text: str,
    raw_transcript: str,
    prompt_type: str,
    *,
    gop_word_scores: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
    cmudict_loader: Any | None = None,
) -> list[dict[str, Any]]:
    expected_words = normalize_for_wer(expected_text).split()
    raw_words = normalize_for_wer(raw_transcript).split()
    pairs = _align_words(expected_words, raw_words)
    gop_by_word = {normalize_for_wer(str(item.get("word", ""))): _optional_float(item.get("score")) for item in gop_word_scores or []}
    alignment: list[dict[str, Any]] = []

    for index, (expected, recognized) in enumerate(pairs):
        if expected is None:
            alignment.append({
                "expected_word": None,
                "recognized_word": recognized,
                "status": "insertion",
                "counts_as_correct": False,
            })
            continue

        recognized_text = recognized or ""
        if not recognized_text:
            alignment.append({
                "expected_word": expected,
                "recognized_word": "",
                "status": "missing",
                "counts_as_correct": False,
            })
            continue

        if expected == recognized_text:
            alignment.append({
                "expected_word": expected,
                "recognized_word": recognized_text,
                "status": "correct",
                "counts_as_correct": True,
            })
            continue

        result = correct_expected_word(
            expected_text=expected,
            raw_transcript=recognized_text,
            prompt_type=prompt_type,
            gop_score=gop_by_word.get(expected),
            retry_required=retry_required,
            uncertain=uncertain,
            context_metadata={"expected_position_context_score": 1.0, "word_index": index},
            cmudict_loader=cmudict_loader,
            config=config,
        )
        status = "incorrect"
        if result["accepted"]:
            status = "accepted_by_homophone" if result["homophone_match"] else "accepted_by_dynamic_expected_word_correction"
        alignment.append({
            "expected_word": expected,
            "recognized_word": recognized_text,
            "status": status,
            "counts_as_correct": bool(result["accepted"]),
            "dynamic_correction_confidence": result["confidence"],
            "spelling_similarity": result["spelling_similarity"],
            "phoneme_similarity": result["phoneme_similarity"],
            "gop_score": result["gop_score"],
            "sub_strategy": result["sub_strategy"],
            "dynamic_correction_reason": result["reason"],
        })

    return alignment


def dynamic_response_fields(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "dynamic_correction_enabled": bool(meta.get("dynamic_correction_enabled", True)),
        "dynamic_correction_applied": bool(meta.get("dynamic_correction_applied", False)),
        "dynamic_correction_strategy": str(meta.get("dynamic_correction_strategy", "dynamic_expected_word_correction")),
        "dynamic_correction_sub_strategy": str(meta.get("dynamic_correction_sub_strategy", "")),
        "dynamic_correction_confidence": meta.get("dynamic_correction_confidence"),
        "dynamic_correction_threshold": meta.get("dynamic_correction_threshold"),
        "dynamic_spelling_similarity": meta.get("dynamic_spelling_similarity"),
        "dynamic_phoneme_similarity": meta.get("dynamic_phoneme_similarity"),
        "dynamic_gop_score": meta.get("dynamic_gop_score"),
        "dynamic_homophone_match": bool(meta.get("dynamic_homophone_match", False)),
        "dynamic_context_score": meta.get("dynamic_context_score"),
        "dynamic_correction_reason": str(meta.get("dynamic_correction_reason", "")),
        "word_alignment": list(meta.get("word_alignment", []) or []),
    }


def _with_dynamic_fields(meta: dict[str, Any], result: DynamicCorrectionResult) -> dict[str, Any]:
    updated = dict(meta)
    updated.update({
        "dynamic_correction_enabled": result.enabled,
        "dynamic_correction_applied": bool(result.accepted),
        "dynamic_correction_strategy": result.strategy,
        "dynamic_correction_sub_strategy": result.sub_strategy,
        "dynamic_correction_confidence": result.confidence,
        "dynamic_correction_threshold": result.threshold,
        "dynamic_spelling_similarity": result.spelling_similarity,
        "dynamic_phoneme_similarity": result.phoneme_similarity,
        "dynamic_gop_score": result.gop_score,
        "dynamic_homophone_match": result.homophone_match,
        "dynamic_context_score": result.context_score,
        "dynamic_correction_reason": result.reason,
    })
    debug_metadata = dict(updated.get("debug_metadata", {}) or {})
    debug_metadata["dynamic_expected_word_correction"] = result.to_dict()
    updated["debug_metadata"] = debug_metadata
    return updated


def _decision(
    *,
    expected: str,
    raw: str,
    prompt_type: str,
    final_score: float,
    threshold: float,
    spelling: float,
    phoneme_score: float | None,
    gop_score: float | None,
    context_score: float,
    known_confusion: float,
    homophone: bool,
    config: dict[str, Any],
) -> tuple[bool, str, str]:
    if _is_unsafe_function_word_pair(expected, raw):
        return False, "rejected_short_function_word_safety", "short function word pair requires stronger exact or acoustic evidence"
    if homophone and context_score >= 0.95:
        return True, "homophone_match", "raw transcript is a homophone or near-identical phoneme match for expected text"
    if known_confusion >= 0.90 and final_score >= min(threshold, 0.78):
        return True, "known_asr_confusion_match", "raw transcript matches a known expected-centric ASR confusion"
    if gop_score is not None and gop_score >= float(config["min_gop_for_acceptance"]) and context_score >= 0.95 and (spelling >= 0.50 or (phoneme_score or 0.0) >= 0.70):
        return True, "gop_supported_expected_match", "GOP and text/context evidence support expected text"
    if phoneme_score is not None and phoneme_score >= float(config["min_phoneme_for_low_text_match"]) and final_score >= threshold:
        return True, "phoneme_expected_match", "raw transcript is close to expected word by phoneme similarity"
    if spelling >= 0.90 and context_score >= 0.95:
        return True, "spelling_context_expected_match", "raw transcript is close to expected word by spelling with strong expected context"
    if phoneme_score is None and gop_score is None and spelling < 0.90:
        return False, "rejected_no_acoustic_evidence", "phoneme/GOP evidence is unavailable and spelling similarity is not high enough"
    if final_score >= threshold and (phoneme_score is not None or gop_score is not None):
        return True, "spelling_phoneme_expected_match", "raw transcript is close to expected word by spelling and phoneme/GOP evidence"
    return False, "rejected_low_similarity", "raw transcript is not close enough to expected text"


def _weighted_score(
    *,
    prompt_type: str,
    spelling_similarity: float,
    phoneme_similarity_score: float | None,
    gop_score: float | None,
    context_score: float,
    known_confusion_score: float,
) -> float:
    if prompt_type == "letter":
        weights = {"phoneme": 0.35, "gop": 0.25, "spelling": 0.20, "known": 0.10, "context": 0.10}
    elif prompt_type in {"sentence", "paragraph", "passage", "reading_passage", "final_sentence", "final_passage"}:
        weights = {"phoneme": 0.30, "gop": 0.20, "spelling": 0.20, "context": 0.20, "known": 0.10}
    else:
        weights = {"phoneme": 0.30, "gop": 0.25, "spelling": 0.20, "context": 0.15, "known": 0.10}
    values = {
        "phoneme": phoneme_similarity_score,
        "gop": gop_score,
        "spelling": spelling_similarity,
        "context": context_score,
        "known": known_confusion_score,
    }
    total_weight = sum(weight for key, weight in weights.items() if values.get(key) is not None)
    if total_weight <= 0:
        return 0.0
    score = sum(weights[key] * float(values[key]) for key in weights if values.get(key) is not None) / total_weight
    return round(max(0.0, min(1.0, score)), 6)


def _align_words(expected: list[str], raw: list[str]) -> list[tuple[str | None, str | None]]:
    n = len(expected)
    m = len(raw)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    move: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
        move[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = float(j)
        move[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = _spelling_similarity(expected[i - 1], raw[j - 1])
            substitution_cost = 0.0 if expected[i - 1] == raw[j - 1] else 1.0 - similarity
            choices = [
                (dp[i - 1][j - 1] + substitution_cost, "substitute"),
                (dp[i - 1][j] + 1.0, "delete"),
                (dp[i][j - 1] + 1.0, "insert"),
            ]
            dp[i][j], move[i][j] = min(choices, key=lambda item: item[0])
    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        action = move[i][j]
        if action == "substitute":
            pairs.append((expected[i - 1], raw[j - 1]))
            i -= 1
            j -= 1
        elif action == "delete":
            pairs.append((expected[i - 1], None))
            i -= 1
        else:
            pairs.append((None, raw[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def _result(
    accepted: bool,
    corrected: str | None,
    display: str,
    prompt_type: str,
    sub_strategy: str,
    confidence: float,
    threshold: float,
    spelling: float,
    phoneme: float | None,
    gop: float | None,
    homophone: bool,
    context: float,
    known: float,
    reason: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    return DynamicCorrectionResult(
        accepted=accepted,
        corrected_text=corrected,
        display_text=display,
        strategy="dynamic_expected_word_correction",
        sub_strategy=sub_strategy,
        confidence=round(confidence, 6),
        threshold=round(threshold, 6),
        spelling_similarity=round(spelling, 6),
        phoneme_similarity=None if phoneme is None else round(phoneme, 6),
        gop_score=None if gop is None else round(gop, 6),
        homophone_match=homophone,
        context_score=round(context, 6),
        known_confusion_score=round(known, 6),
        reason=reason,
        enabled=enabled,
    ).to_dict()


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_DYNAMIC_CONFIG, **(config or {})}
    return {
        "enabled": _as_bool(merged.get("enabled", True)),
        "letter_accept_threshold": float(merged.get("letter_accept_threshold", merged.get("letter_threshold", 0.72))),
        "word_accept_threshold": float(merged.get("word_accept_threshold", merged.get("word_threshold", 0.78))),
        "rhyme_accept_threshold": float(merged.get("rhyme_accept_threshold", merged.get("rhyme_threshold", 0.78))),
        "sentence_word_accept_threshold": float(merged.get("sentence_word_accept_threshold", 0.80)),
        "passage_word_accept_threshold": float(merged.get("passage_word_accept_threshold", 0.82)),
        "homophone_threshold": float(merged.get("homophone_threshold", 0.96)),
        "min_phoneme_for_low_text_match": float(merged.get("min_phoneme_for_low_text_match", 0.90)),
        "min_gop_for_acceptance": float(merged.get("min_gop_for_acceptance", 0.75)),
        "skip_on_retry_required": _as_bool(merged.get("skip_on_retry_required", True)),
        "skip_on_uncertain_audio": _as_bool(merged.get("skip_on_uncertain_audio", True)),
        "debug": _as_bool(merged.get("debug", False)),
    }


def _threshold_for_prompt(prompt_type: str, config: dict[str, Any]) -> float:
    if prompt_type == "letter":
        return float(config["letter_accept_threshold"])
    if prompt_type in {"rhyme", "rhyming_word"}:
        return float(config["rhyme_accept_threshold"])
    if prompt_type in {"sentence", "final_sentence"}:
        return float(config["sentence_word_accept_threshold"])
    if prompt_type in {"paragraph", "passage", "reading_passage", "final_passage"}:
        return float(config["passage_word_accept_threshold"])
    return float(config["word_accept_threshold"])


def _spelling_similarity(expected: str, raw: str) -> float:
    left = expected.replace("'", "")
    right = raw.replace("'", "")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 6)


def _phoneme_similarity(expected: str, raw: str, cmudict_loader: Any | None) -> float | None:
    expected_phonemes, _, _ = generate_expected_phonemes(expected, cmudict_loader)
    raw_phonemes, _, _ = generate_expected_phonemes(raw, cmudict_loader)
    if not expected_phonemes or not raw_phonemes:
        return None
    return phoneme_similarity(expected_phonemes, raw_phonemes)


def _known_confusion_score(expected: str, raw: str, prompt_type: str) -> float:
    if expected == raw:
        return 1.0
    raw_no_apostrophe = raw.replace("'", "")
    expected_no_apostrophe = expected.replace("'", "")
    if raw_no_apostrophe in KNOWN_ASR_CONFUSIONS.get(expected_no_apostrophe, set()):
        return 0.95
    if prompt_type == "letter" and _letter_alias_match(expected, raw):
        return 0.95
    return 0.0


def _letter_alias_match(expected: str, raw: str) -> bool:
    if len(expected) != 1 or not expected.isalpha():
        return False
    forms = {normalize_for_wer(form).replace(" ", "") for form in LETTER_PRONUNCIATIONS.get(expected.lower(), set())}
    forms.update({"you"} if expected.lower() == "q" else set())
    compact_raw = normalize_for_wer(raw).replace(" ", "")
    return compact_raw in forms


def _exact_match(expected: str, raw: str) -> bool:
    return bool(expected) and expected == raw


def _context_score(context: dict[str, Any]) -> float:
    value = context.get("expected_position_context_score", context.get("context_score", 1.0))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_unsafe_function_word_pair(expected: str, raw: str) -> bool:
    if expected == raw:
        return False
    return expected in FUNCTION_WORDS and raw in FUNCTION_WORDS


def _critical_phoneme_contradicted(meta: dict[str, Any]) -> bool:
    return bool(meta.get("critical_pair_detected", False)) and meta.get("critical_phoneme_detected") is False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
