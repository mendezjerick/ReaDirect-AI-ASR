from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer
from readirect_asr.scoring.phoneme_comparison import phoneme_similarity as phoneme_sequence_similarity
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
    "fragment_detection_enabled": True,
    "reject_consonant_fragment_by_default": True,
    "min_raw_length_ratio_for_words": 0.50,
    "min_phoneme_coverage_for_words": 0.65,
    "fragment_gop_accept_threshold": 0.88,
    "fragment_phoneme_accept_threshold": 0.82,
    "fragment_retry_on_bad_audio": True,
    "fragment_allow_accept_with_strong_gop": True,
    "asr_spelling_variant_enabled": True,
    "asr_variant_word_accept_threshold": 0.78,
    "asr_variant_rhyme_accept_threshold": 0.78,
    "asr_variant_letter_accept_threshold": 0.72,
    "asr_variant_sentence_word_accept_threshold": 0.82,
    "asr_variant_passage_word_accept_threshold": 0.84,
    "asr_variant_min_consonant_skeleton": 0.80,
    "asr_variant_min_vowel_tolerant_score": 0.78,
    "asr_variant_min_gop_for_risky_accept": 0.82,
    "asr_variant_min_phoneme_coverage": 0.70,
    "asr_variant_short_word_strict": True,
    "asr_variant_debug": True,
    "dynamic_alignment_repair_enabled": True,
    "dynamic_alignment_allow_split_merge": True,
    "dynamic_alignment_max_expected_chunk": 3,
    "dynamic_alignment_max_raw_chunk": 3,
    "dynamic_alignment_accept_threshold": 0.82,
    "dynamic_alignment_partial_threshold": 0.65,
    "dynamic_alignment_split_merge_threshold": 0.84,
    "dynamic_alignment_homophone_threshold": 0.96,
    "dynamic_alignment_gop_accept_threshold": 0.75,
    "dynamic_alignment_short_function_word_strict": True,
    "dynamic_alignment_debug": True,
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
    suspicious_fragment: bool = False
    fragment_reasons: list[str] | None = None
    phoneme_coverage: float | None = None
    asr_spelling_variant_enabled: bool = True
    asr_spelling_variant_applied: bool = False
    asr_spelling_variant_strategy: str = "dynamic_asr_spelling_variant"
    asr_spelling_variant_sub_strategy: str = ""
    asr_spelling_variant_confidence: float | None = None
    asr_spelling_variant_threshold: float | None = None
    consonant_skeleton_similarity: float | None = None
    vowel_tolerant_similarity: float | None = None
    expected_phoneme_coverage: float | None = None
    variant_edit_similarity: float | None = None
    variant_reason: str = ""

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
            "suspicious_fragment": self.suspicious_fragment,
            "fragment_reasons": self.fragment_reasons or [],
            "phoneme_coverage": self.phoneme_coverage,
            "asr_spelling_variant_enabled": self.asr_spelling_variant_enabled,
            "asr_spelling_variant_applied": self.asr_spelling_variant_applied,
            "asr_spelling_variant_strategy": self.asr_spelling_variant_strategy,
            "asr_spelling_variant_sub_strategy": self.asr_spelling_variant_sub_strategy,
            "asr_spelling_variant_confidence": self.asr_spelling_variant_confidence,
            "asr_spelling_variant_threshold": self.asr_spelling_variant_threshold,
            "consonant_skeleton_similarity": self.consonant_skeleton_similarity,
            "vowel_tolerant_similarity": self.vowel_tolerant_similarity,
            "expected_phoneme_coverage": self.expected_phoneme_coverage,
            "variant_edit_similarity": self.variant_edit_similarity,
            "variant_reason": self.variant_reason,
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
        fragment = _short_word_fragment(
            expected=normalized_expected,
            raw=normalized_raw,
            prompt_type=detected_prompt_type,
            phoneme_score=None,
            audio_quality=audio_quality,
            config=active_config,
        )
        return _result(
            False,
            None,
            raw,
            detected_prompt_type,
            "rejected_blank_raw_transcript",
            0.0,
            threshold,
            0.0,
            None,
            gop_score,
            False,
            context_score,
            0.0,
            "raw transcript is blank",
            suspicious_fragment=fragment["suspicious_fragment"],
            fragment_reasons=fragment["fragment_reasons"],
            phoneme_coverage=fragment["phoneme_coverage"],
        )

    spelling = _spelling_similarity(normalized_expected, normalized_raw)
    phoneme_score = phoneme_similarity_score
    if phoneme_score is None:
        phoneme_score = _phoneme_similarity(normalized_expected, normalized_raw, cmudict_loader)
    known_confusion = _known_confusion_score(normalized_expected, normalized_raw, detected_prompt_type)
    homophone = bool(phoneme_score is not None and phoneme_score >= float(active_config["homophone_threshold"]) and normalized_expected != normalized_raw)
    fragment = _short_word_fragment(
        expected=normalized_expected,
        raw=normalized_raw,
        prompt_type=detected_prompt_type,
        phoneme_score=phoneme_score,
        audio_quality=audio_quality,
        config=active_config,
    )

    if _exact_match(normalized_expected, normalized_raw):
        return _result(True, expected, expected, detected_prompt_type, "exact_normalized_match", 1.0, threshold, 1.0, phoneme_score, gop_score, False, context_score, known_confusion, "raw transcript matches expected text after normalization")

    if detected_prompt_type == "letter" and _letter_alias_match(normalized_expected, normalized_raw):
        return _result(True, expected, expected, detected_prompt_type, "letter_alias_match", 0.98, threshold, spelling, phoneme_score, gop_score, False, context_score, max(known_confusion, 0.9), "raw transcript is a spoken form of the expected letter")

    if fragment["suspicious_fragment"]:
        strong_gop = (
            bool(active_config["fragment_allow_accept_with_strong_gop"])
            and gop_score is not None
            and gop_score >= float(active_config["fragment_gop_accept_threshold"])
        )
        strong_phoneme = phoneme_score is not None and phoneme_score >= float(active_config["fragment_phoneme_accept_threshold"])
        fragment_confidence = max(gop_score or 0.0, phoneme_score or 0.0)
        fragment_reason = "suspicious_fragment: " + ", ".join(fragment["fragment_reasons"])

        if bool(active_config["fragment_retry_on_bad_audio"]) and fragment["bad_audio"]:
            return _result(
                False,
                None,
                raw,
                detected_prompt_type,
                "fragment_retry_bad_audio",
                fragment_confidence,
                float(active_config["fragment_gop_accept_threshold"]),
                spelling,
                phoneme_score,
                gop_score,
                homophone,
                context_score,
                known_confusion,
                fragment_reason + "; audio was not reliable enough for correction",
                suspicious_fragment=True,
                fragment_reasons=fragment["fragment_reasons"],
                phoneme_coverage=fragment["phoneme_coverage"],
            )

        if strong_gop or strong_phoneme:
            sub_strategy = "fragment_gop_supported_expected_match" if strong_gop else "fragment_phoneme_supported_expected_match"
            return _result(
                True,
                expected,
                expected,
                detected_prompt_type,
                sub_strategy,
                fragment_confidence,
                float(active_config["fragment_gop_accept_threshold"] if strong_gop else active_config["fragment_phoneme_accept_threshold"]),
                spelling,
                phoneme_score,
                gop_score,
                homophone,
                context_score,
                known_confusion,
                fragment_reason + "; strong pronunciation evidence supports the expected word",
                suspicious_fragment=True,
                fragment_reasons=fragment["fragment_reasons"],
                phoneme_coverage=fragment["phoneme_coverage"],
            )

        return _result(
            False,
            None,
            raw,
            detected_prompt_type,
            "rejected_suspicious_fragment_low_pronunciation_evidence",
            fragment_confidence,
            float(active_config["fragment_gop_accept_threshold"]),
            spelling,
            phoneme_score,
            gop_score,
            homophone,
            context_score,
            known_confusion,
            fragment_reason + "; missing strong GOP/phoneme evidence for the expected vowel or ending sounds",
            suspicious_fragment=True,
            fragment_reasons=fragment["fragment_reasons"],
            phoneme_coverage=fragment["phoneme_coverage"],
        )

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

    variant_result: dict[str, Any] | None = None
    can_prefer_variant = sub_strategy not in {"homophone_match", "known_asr_confusion_match", "gop_supported_expected_match"}
    if not accepted or can_prefer_variant:
        variant_result = detect_asr_spelling_variant(
            expected_text=expected,
            raw_transcript=raw,
            prompt_type=detected_prompt_type,
            gop_score=gop_score,
            phoneme_similarity=phoneme_score,
            audio_quality=audio_quality,
            retry_required=retry_required,
            uncertain=uncertain,
            context_metadata=context,
            config=active_config,
        )
        if variant_result["accepted"] and can_prefer_variant:
            return _result(
                accepted=True,
                corrected=expected,
                display=expected,
                prompt_type=detected_prompt_type,
                sub_strategy=variant_result["sub_strategy"],
                confidence=variant_result["confidence"],
                threshold=variant_result["threshold"],
                spelling=spelling,
                phoneme=phoneme_score,
                gop=gop_score,
                homophone=homophone,
                context=context_score,
                known=known_confusion,
                reason=variant_result["reason"],
                strategy="dynamic_asr_spelling_variant",
                variant=variant_result,
            )

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
        variant=variant_result,
    )


def detect_asr_spelling_variant(
    expected_text: str,
    raw_transcript: str,
    prompt_type: str,
    gop_score: float | None = None,
    phoneme_similarity: float | None = None,
    expected_phonemes: list[str] | None = None,
    observed_phonemes: list[str] | None = None,
    audio_quality: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
    context_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_config = _config(config)
    detected_prompt_type = detect_prompt_type(expected_text, prompt_type=prompt_type)
    threshold = _variant_threshold_for_prompt(detected_prompt_type, active_config)
    expected = normalize_for_wer(str(expected_text or ""))
    raw = normalize_for_wer(str(raw_transcript or ""))
    context_score = _context_score(context_metadata or {})
    phoneme_coverage = phoneme_similarity

    if expected_phonemes and observed_phonemes and phoneme_coverage is None:
        phoneme_coverage = phoneme_sequence_similarity(expected_phonemes, observed_phonemes)

    empty = _variant_result(False, None, raw_transcript, "skipped", 0.0, threshold, 0.0, 0.0, 0.0, phoneme_similarity, gop_score, phoneme_coverage, context_score, "not evaluated")

    if not bool(active_config["asr_spelling_variant_enabled"]):
        return {**empty, "sub_strategy": "disabled", "reason": "ASR spelling-variant correction is disabled"}
    if retry_required:
        return {**empty, "sub_strategy": "skipped_retry_required", "reason": "retry_required is true"}
    if uncertain and bool(active_config["skip_on_uncertain_audio"]):
        return {**empty, "sub_strategy": "skipped_uncertain_audio", "reason": "uncertain audio is true"}
    if not expected:
        return {**empty, "sub_strategy": "skipped_no_expected_text", "reason": "expected_text is missing"}
    if not raw:
        return {**empty, "sub_strategy": "rejected_blank_raw_transcript", "reason": "raw transcript is blank"}
    if expected == raw:
        return {**empty, "sub_strategy": "exact_match_not_variant", "reason": "raw transcript already matches expected text"}
    if _is_unsafe_function_word_pair(expected, raw):
        return {**empty, "sub_strategy": "rejected_short_function_word_safety", "reason": "short function word mismatch is too risky for variant correction"}
    if _bad_audio_for_fragment(audio_quality):
        return {**empty, "sub_strategy": "skipped_bad_audio", "reason": "audio quality is not reliable enough for spelling-variant correction"}

    edit_similarity = _spelling_similarity(expected, raw)
    skeleton_similarity = _consonant_skeleton_similarity(expected, raw)
    vowel_similarity = _vowel_tolerant_similarity(expected, raw)
    score = _variant_weighted_score(
        prompt_type=detected_prompt_type,
        consonant_skeleton_similarity=skeleton_similarity,
        vowel_tolerant_similarity=vowel_similarity,
        gop_score=gop_score,
        expected_phoneme_coverage=phoneme_coverage,
        edit_similarity=edit_similarity,
        context_score=context_score,
    )
    close_enough = (
        skeleton_similarity >= float(active_config["asr_variant_min_consonant_skeleton"])
        or edit_similarity >= 0.84
        or (gop_score or 0.0) >= float(active_config["asr_variant_min_gop_for_risky_accept"])
        or (phoneme_coverage or 0.0) >= float(active_config["asr_variant_min_phoneme_coverage"])
    )
    vowel_or_evidence = (
        vowel_similarity >= float(active_config["asr_variant_min_vowel_tolerant_score"])
        or (gop_score or 0.0) >= float(active_config["asr_variant_min_gop_for_risky_accept"])
        or (phoneme_coverage or 0.0) >= float(active_config["asr_variant_min_phoneme_coverage"])
    )
    risky_short = bool(active_config["asr_variant_short_word_strict"]) and len(expected) <= 3 and (gop_score is None and phoneme_coverage is None)
    accepted = bool(score >= threshold and close_enough and vowel_or_evidence and not risky_short)
    sub_strategy = "vowel_tolerant_consonant_skeleton_match" if accepted else "rejected_low_variant_confidence"
    reason = "raw transcript appears to be a noisy ASR spelling of the expected word" if accepted else "raw transcript is not close enough to expected word"

    return _variant_result(
        accepted,
        expected_text if accepted else None,
        expected_text if accepted else raw_transcript,
        sub_strategy,
        score,
        threshold,
        skeleton_similarity,
        vowel_similarity,
        edit_similarity,
        phoneme_similarity,
        gop_score,
        phoneme_coverage,
        context_score,
        reason,
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
        accepted_statuses = {
            "exact_correct",
            "accepted_by_dynamic_expected_word_correction",
            "accepted_by_homophone",
            "accepted_by_phoneme_similarity",
            "accepted_by_gop",
            "accepted_by_asr_spelling_variant",
            "accepted_by_split_merge",
        }
        applied_items = [item for item in alignment if item.get("status") in accepted_statuses and item.get("status") != "exact_correct"]
        best = max((float(item.get("alignment_confidence", item.get("dynamic_correction_confidence", 0.0)) or 0.0) for item in applied_items), default=0.0)
        result = _result(
            accepted=bool(applied_items),
            corrected=None,
            display=raw,
            prompt_type=detected_prompt_type,
            sub_strategy="passage_aware_dynamic_alignment_repair" if any(item.get("status") == "accepted_by_split_merge" for item in applied_items) else ("word_alignment_dynamic_correction" if applied_items else "no_word_alignment_corrections"),
            confidence=best,
            threshold=_threshold_for_prompt(detected_prompt_type, _config(config)),
            spelling=0.0,
            phoneme=None,
            gop=gop_score,
            homophone=any(item.get("status") == "accepted_by_homophone" for item in applied_items),
            context=_context_score(context),
            known=0.0,
            reason="passage-aware alignment repair applied to word chunks only" if applied_items else "no aligned word met dynamic correction threshold",
        )
        updated = _with_dynamic_fields(updated, DynamicCorrectionResult(**result))
        updated["word_alignment"] = alignment
        debug_metadata = dict(updated.get("debug_metadata", {}) or {})
        debug_metadata["word_alignment"] = alignment
        debug_metadata["dynamic_expected_word_correction"] = result
        if bool(_config(config)["dynamic_alignment_debug"]):
            debug_metadata["alignment_debug"] = _alignment_debug(expected, raw, alignment)
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
        if result.get("strategy") == "dynamic_asr_spelling_variant":
            updated["correction_strategy_used"] = "dynamic_asr_spelling_variant"
        elif result.get("sub_strategy") == "fragment_gop_supported_expected_match":
            updated["correction_strategy_used"] = "dynamic_expected_word_fragment_gop_support"
        elif result.get("sub_strategy") == "fragment_phoneme_supported_expected_match":
            updated["correction_strategy_used"] = "dynamic_expected_word_fragment_phoneme_support"
        else:
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
    active_config = _config(config)
    if bool(active_config["dynamic_alignment_repair_enabled"]):
        return _dynamic_chunk_alignment(
            expected_words,
            raw_words,
            prompt_type,
            gop_word_scores=gop_word_scores or [],
            config=active_config,
            retry_required=retry_required,
            uncertain=uncertain,
            cmudict_loader=cmudict_loader,
        )

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
                "status": "exact_correct",
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
            if result.get("strategy") == "dynamic_asr_spelling_variant":
                status = "accepted_by_asr_spelling_variant"
            else:
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
            "asr_spelling_variant_applied": result.get("asr_spelling_variant_applied", False),
            "asr_spelling_variant_confidence": result.get("asr_spelling_variant_confidence"),
            "consonant_skeleton_similarity": result.get("consonant_skeleton_similarity"),
            "vowel_tolerant_similarity": result.get("vowel_tolerant_similarity"),
            "expected_phoneme_coverage": result.get("expected_phoneme_coverage"),
            "variant_edit_similarity": result.get("variant_edit_similarity"),
            "variant_reason": result.get("variant_reason", ""),
        })

    return alignment


def _dynamic_chunk_alignment(
    expected_words: list[str],
    raw_words: list[str],
    prompt_type: str,
    *,
    gop_word_scores: list[dict[str, Any]],
    config: dict[str, Any],
    retry_required: bool,
    uncertain: bool,
    cmudict_loader: Any | None,
) -> list[dict[str, Any]]:
    n = len(expected_words)
    m = len(raw_words)
    gop_by_word = {normalize_for_wer(str(item.get("word", ""))): _optional_float(item.get("score")) for item in gop_word_scores}
    phoneme_cache: dict[tuple[str, str], float | None] = {}
    dynamic_result_cache: dict[tuple[str, str], dict[str, Any]] = {}
    inf = 1_000_000.0
    dp: list[list[tuple[float, list[dict[str, Any]]]]] = [[(inf, []) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = (0.0, [])

    for i in range(n + 1):
        for j in range(m + 1):
            base_cost, base_ops = dp[i][j]
            if base_cost >= inf:
                continue

            if i < n:
                op = _missing_operation(expected_words[i])
                _relax(dp, i + 1, j, base_cost + 1.0, base_ops + [op])

            if j < m:
                op = _inserted_operation(raw_words[j])
                _relax(dp, i, j + 1, base_cost + 0.95, base_ops + [op])

            for expected_count, raw_count in _candidate_chunk_sizes(config):
                if expected_count <= 0 or raw_count <= 0 or i + expected_count > n or j + raw_count > m:
                    continue
                expected_chunk = expected_words[i:i + expected_count]
                raw_chunk = raw_words[j:j + raw_count]
                candidate = _score_alignment_candidate(
                    expected_chunk,
                    raw_chunk,
                    prompt_type,
                    gop_by_word=gop_by_word,
                    config=config,
                    retry_required=retry_required,
                    uncertain=uncertain,
                    cmudict_loader=cmudict_loader,
                    phoneme_cache=phoneme_cache,
                    dynamic_result_cache=dynamic_result_cache,
                )
                if expected_count == 3 or raw_count == 3:
                    if not _is_obvious_three_word_boundary_repair(candidate, config):
                        continue
                _relax(
                    dp,
                    i + expected_count,
                    j + raw_count,
                    base_cost + _candidate_cost(candidate, config),
                    base_ops + [candidate],
                )

    operations = dp[n][m][1]
    alignment: list[dict[str, Any]] = []
    chunk_index = 1
    for operation in operations:
        alignment.extend(_expand_operation(operation, chunk_index))
        if operation.get("expected_words") or operation.get("recognized_words"):
            chunk_index += 1
    return alignment


def _candidate_chunk_sizes(config: dict[str, Any]) -> list[tuple[int, int]]:
    if not bool(config["dynamic_alignment_allow_split_merge"]):
        return [(1, 1)]
    max_expected = max(1, min(3, int(config["dynamic_alignment_max_expected_chunk"])))
    max_raw = max(1, min(3, int(config["dynamic_alignment_max_raw_chunk"])))
    candidates: list[tuple[int, int]] = []
    for expected_count, raw_count in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]:
        if expected_count <= max_expected and raw_count <= max_raw:
            candidates.append((expected_count, raw_count))
    return candidates


def _score_alignment_candidate(
    expected_words: list[str],
    raw_words: list[str],
    prompt_type: str,
    *,
    gop_by_word: dict[str, float | None],
    config: dict[str, Any],
    retry_required: bool,
    uncertain: bool,
    cmudict_loader: Any | None,
    phoneme_cache: dict[tuple[str, str], float | None],
    dynamic_result_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    expected_spaced = " ".join(expected_words)
    raw_spaced = " ".join(raw_words)
    expected_joined = "".join(expected_words)
    raw_joined = "".join(raw_words)
    expected_count = len(expected_words)
    raw_count = len(raw_words)
    operation = _operation_name(expected_count, raw_count)
    gop_score = _mean_optional([gop_by_word.get(word) for word in expected_words])
    spelling_similarity = max(_spelling_similarity(expected_spaced, raw_spaced), _spelling_similarity(expected_joined, raw_joined))
    boundary_similarity = _spelling_similarity(expected_joined, raw_joined)
    vowel_similarity = _vowel_tolerant_similarity(expected_joined, raw_joined)
    skeleton_similarity = _consonant_skeleton_similarity(expected_joined, raw_joined)
    context_score = 1.0
    is_split_merge = expected_count != raw_count
    exact_match = expected_spaced == raw_spaced
    cheap_similarity = max(spelling_similarity, boundary_similarity, vowel_similarity, skeleton_similarity)
    should_compute_phonemes = (
        not exact_match
        and (
            cheap_similarity >= 0.58
            or (is_split_merge and boundary_similarity >= 0.62)
            or (gop_score is not None and gop_score >= float(config["dynamic_alignment_gop_accept_threshold"]))
        )
    )
    phoneme_similarity = (
        _cached_phoneme_similarity(expected_spaced, raw_spaced, cmudict_loader, phoneme_cache)
        if should_compute_phonemes
        else None
    )
    homophone = bool(phoneme_similarity is not None and phoneme_similarity >= float(config["dynamic_alignment_homophone_threshold"]) and expected_joined != raw_joined)

    dynamic_result: dict[str, Any] | None = None
    if expected_count == 1 and raw_count == 1 and not exact_match and should_compute_phonemes:
        dynamic_key = (expected_spaced, raw_spaced)
        if dynamic_key not in dynamic_result_cache:
            dynamic_result_cache[dynamic_key] = correct_expected_word(
                expected_text=expected_spaced,
                raw_transcript=raw_spaced,
                prompt_type=prompt_type,
                gop_score=gop_score,
                phoneme_similarity_score=phoneme_similarity,
                retry_required=retry_required,
                uncertain=uncertain,
                context_metadata={"expected_position_context_score": context_score},
                cmudict_loader=cmudict_loader,
                config=config,
            )
        dynamic_result = dynamic_result_cache[dynamic_key]

    confidence = _alignment_confidence(
        spelling_similarity=spelling_similarity,
        boundary_similarity=boundary_similarity,
        skeleton_similarity=skeleton_similarity,
        vowel_similarity=vowel_similarity,
        phoneme_similarity=phoneme_similarity,
        gop_score=gop_score,
        context_score=context_score,
        is_split_merge=is_split_merge,
    )
    status = "incorrect"
    reason = "candidate did not meet passage-aware alignment thresholds"
    counts_as_correct = False
    sub_strategy = "rejected_low_alignment_confidence"
    unsafe_function_pair = (
        bool(config["dynamic_alignment_short_function_word_strict"])
        and expected_count == 1
        and raw_count == 1
        and _is_unsafe_function_word_pair(expected_spaced, raw_spaced)
    )

    if exact_match:
        status = "exact_correct"
        counts_as_correct = True
        confidence = 1.0
        reason = "expected and raw word match exactly"
        sub_strategy = "exact_normalized_match"
    elif unsafe_function_pair:
        status = "incorrect"
        reason = "short function word substitution is too risky for alignment repair"
        confidence = min(confidence, 0.49)
    elif dynamic_result and dynamic_result.get("accepted"):
        counts_as_correct = True
        confidence = max(confidence, float(dynamic_result.get("confidence", 0.0) or 0.0))
        sub_strategy = str(dynamic_result.get("sub_strategy", "dynamic_expected_word_correction"))
        reason = str(dynamic_result.get("reason", "dynamic expected-word correction accepted the aligned word"))
        if dynamic_result.get("strategy") == "dynamic_asr_spelling_variant":
            status = "accepted_by_asr_spelling_variant"
        else:
            status = "accepted_by_homophone" if dynamic_result.get("homophone_match") else "accepted_by_dynamic_expected_word_correction"
    elif homophone:
        status = "accepted_by_homophone"
        counts_as_correct = True
        confidence = max(confidence, phoneme_similarity or 0.0)
        reason = "expected and raw chunks are homophones or near-identical phoneme matches"
        sub_strategy = "homophone_match"
    elif is_split_merge and _split_merge_accepts(boundary_similarity, confidence, phoneme_similarity, gop_score, skeleton_similarity, vowel_similarity, config):
        status = "accepted_by_split_merge"
        counts_as_correct = True
        reason = "expected and raw chunks match after repairing word boundaries"
        sub_strategy = "boundary_repair_expected_raw_joined_match"
    elif gop_score is not None and gop_score >= float(config["dynamic_alignment_gop_accept_threshold"]) and confidence >= float(config["dynamic_alignment_accept_threshold"]):
        status = "accepted_by_gop"
        counts_as_correct = True
        confidence = max(confidence, gop_score)
        reason = "GOP word score and alignment evidence support the expected chunk"
        sub_strategy = "gop_supported_chunk_match"
    elif phoneme_similarity is not None and phoneme_similarity >= float(config["dynamic_alignment_accept_threshold"]) and confidence >= float(config["dynamic_alignment_accept_threshold"]):
        status = "accepted_by_phoneme_similarity"
        counts_as_correct = True
        confidence = max(confidence, phoneme_similarity)
        reason = "phoneme similarity supports the expected chunk"
        sub_strategy = "phoneme_supported_chunk_match"
    elif confidence >= float(config["dynamic_alignment_partial_threshold"]):
        status = "partial"
        reason = "candidate has partial alignment evidence but not enough to count as correct"
        sub_strategy = "partial_chunk_match"

    return {
        "expected_words": expected_words,
        "recognized_words": raw_words,
        "expected_chunk": expected_spaced,
        "recognized_chunk": raw_spaced,
        "operation": operation,
        "status": status,
        "counts_as_correct": counts_as_correct,
        "alignment_confidence": round(confidence, 6),
        "spelling_similarity": round(spelling_similarity, 6),
        "boundary_similarity": round(boundary_similarity, 6),
        "phoneme_similarity": None if phoneme_similarity is None else round(phoneme_similarity, 6),
        "gop_score": None if gop_score is None else round(gop_score, 6),
        "consonant_skeleton_similarity": round(skeleton_similarity, 6),
        "vowel_tolerant_similarity": round(vowel_similarity, 6),
        "sub_strategy": sub_strategy,
        "dynamic_correction_reason": reason,
        "dynamic_correction_confidence": round(confidence, 6),
        "asr_spelling_variant_applied": bool(dynamic_result and dynamic_result.get("asr_spelling_variant_applied")),
        "asr_spelling_variant_confidence": dynamic_result.get("asr_spelling_variant_confidence") if dynamic_result else None,
        "variant_reason": dynamic_result.get("variant_reason", "") if dynamic_result else "",
    }


def _alignment_confidence(
    *,
    spelling_similarity: float,
    boundary_similarity: float,
    skeleton_similarity: float,
    vowel_similarity: float,
    phoneme_similarity: float | None,
    gop_score: float | None,
    context_score: float,
    is_split_merge: bool,
) -> float:
    if is_split_merge:
        weights = {
            "boundary": 0.28,
            "skeleton": 0.17,
            "vowel": 0.15,
            "phoneme": 0.25,
            "gop": 0.10,
            "context": 0.05,
        }
        values = {
            "boundary": boundary_similarity,
            "skeleton": skeleton_similarity,
            "vowel": vowel_similarity,
            "phoneme": phoneme_similarity,
            "gop": gop_score,
            "context": context_score,
        }
    else:
        weights = {
            "spelling": 0.25,
            "skeleton": 0.15,
            "vowel": 0.15,
            "phoneme": 0.30,
            "gop": 0.10,
            "context": 0.05,
        }
        values = {
            "spelling": spelling_similarity,
            "skeleton": skeleton_similarity,
            "vowel": vowel_similarity,
            "phoneme": phoneme_similarity,
            "gop": gop_score,
            "context": context_score,
        }

    total = sum(weight for key, weight in weights.items() if values.get(key) is not None)
    if total <= 0:
        return 0.0
    return round(sum(weights[key] * float(values[key]) for key in weights if values.get(key) is not None) / total, 6)


def _split_merge_accepts(
    boundary_similarity: float,
    confidence: float,
    phoneme_similarity: float | None,
    gop_score: float | None,
    skeleton_similarity: float,
    vowel_similarity: float,
    config: dict[str, Any],
) -> bool:
    if boundary_similarity >= float(config["dynamic_alignment_split_merge_threshold"]):
        return True
    if confidence < float(config["dynamic_alignment_accept_threshold"]):
        return False
    return (
        (phoneme_similarity or 0.0) >= 0.80
        or (gop_score or 0.0) >= float(config["dynamic_alignment_gop_accept_threshold"])
        or (skeleton_similarity >= 0.95 and vowel_similarity >= 0.70)
    )


def _candidate_cost(candidate: dict[str, Any], config: dict[str, Any]) -> float:
    status = str(candidate.get("status", "incorrect"))
    confidence = float(candidate.get("alignment_confidence", 0.0) or 0.0)
    if status == "exact_correct":
        return 0.01
    if status.startswith("accepted_by_"):
        return max(0.04, 0.35 - (confidence * 0.25))
    if status == "partial":
        return max(0.60, 1.0 - (confidence * 0.35))
    return 1.05


def _operation_name(expected_count: int, raw_count: int) -> str:
    if expected_count == 1 and raw_count == 1:
        return "match"
    if expected_count == 1 and raw_count > 1:
        return "split_match"
    if expected_count > 1 and raw_count == 1:
        return "merge_match"
    return "phrase_match"


def _missing_operation(expected_word: str) -> dict[str, Any]:
    return {
        "expected_words": [expected_word],
        "recognized_words": [],
        "expected_chunk": expected_word,
        "recognized_chunk": "",
        "operation": "deletion",
        "status": "missing",
        "counts_as_correct": False,
        "alignment_confidence": 0.0,
        "dynamic_correction_reason": "expected word was not aligned to a raw token",
    }


def _inserted_operation(raw_word: str) -> dict[str, Any]:
    return {
        "expected_words": [],
        "recognized_words": [raw_word],
        "expected_chunk": "",
        "recognized_chunk": raw_word,
        "operation": "insertion",
        "status": "inserted",
        "counts_as_correct": False,
        "alignment_confidence": 0.0,
        "dynamic_correction_reason": "raw token was inserted relative to expected passage",
    }


def _expand_operation(operation: dict[str, Any], chunk_index: int) -> list[dict[str, Any]]:
    expected_words = list(operation.get("expected_words", []) or [])
    recognized_words = list(operation.get("recognized_words", []) or [])
    recognized_chunk = str(operation.get("recognized_chunk", ""))
    expected_chunk = str(operation.get("expected_chunk", ""))
    chunk_id = f"chunk_{chunk_index:03d}"

    if not expected_words:
        return [{
            **_public_alignment_fields(operation, chunk_id),
            "expected_word": None,
            "recognized_word": recognized_chunk,
        }]

    return [
        {
            **_public_alignment_fields(operation, chunk_id),
            "expected_word": expected_word,
            "recognized_word": recognized_chunk if len(recognized_words) != len(expected_words) else (recognized_words[index] if index < len(recognized_words) else recognized_chunk),
        }
        for index, expected_word in enumerate(expected_words)
    ]


def _public_alignment_fields(operation: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return {
        "expected_chunk": operation.get("expected_chunk", ""),
        "recognized_chunk": operation.get("recognized_chunk", ""),
        "chunk_match_id": chunk_id,
        "operation": operation.get("operation", ""),
        "status": operation.get("status", "incorrect"),
        "counts_as_correct": bool(operation.get("counts_as_correct", False)),
        "alignment_confidence": operation.get("alignment_confidence"),
        "dynamic_correction_confidence": operation.get("dynamic_correction_confidence", operation.get("alignment_confidence")),
        "spelling_similarity": operation.get("spelling_similarity"),
        "boundary_similarity": operation.get("boundary_similarity"),
        "phoneme_similarity": operation.get("phoneme_similarity"),
        "gop_score": operation.get("gop_score"),
        "sub_strategy": operation.get("sub_strategy"),
        "dynamic_correction_reason": operation.get("dynamic_correction_reason"),
        "asr_spelling_variant_applied": operation.get("asr_spelling_variant_applied", False),
        "asr_spelling_variant_confidence": operation.get("asr_spelling_variant_confidence"),
        "consonant_skeleton_similarity": operation.get("consonant_skeleton_similarity"),
        "vowel_tolerant_similarity": operation.get("vowel_tolerant_similarity"),
        "variant_reason": operation.get("variant_reason", ""),
    }


def _is_obvious_three_word_boundary_repair(candidate: dict[str, Any], config: dict[str, Any]) -> bool:
    return (
        candidate.get("status") == "accepted_by_split_merge"
        and (
            float(candidate.get("boundary_similarity", 0.0) or 0.0) >= 0.92
            or float(candidate.get("phoneme_similarity", 0.0) or 0.0) >= 0.90
            or float(candidate.get("alignment_confidence", 0.0) or 0.0) >= 0.90
        )
    )


def _relax(
    dp: list[list[tuple[float, list[dict[str, Any]]]]],
    i: int,
    j: int,
    cost: float,
    operations: list[dict[str, Any]],
) -> None:
    if cost < dp[i][j][0]:
        dp[i][j] = (cost, operations)


def _mean_optional(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _cached_phoneme_similarity(
    expected: str,
    raw: str,
    cmudict_loader: Any | None,
    cache: dict[tuple[str, str], float | None],
) -> float | None:
    key = (expected, raw)
    if key not in cache:
        cache[key] = _phoneme_similarity(expected, raw, cmudict_loader)
    return cache[key]


def _alignment_debug(expected_text: str, raw_transcript: str, alignment: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_statuses = {
        "exact_correct",
        "accepted_by_dynamic_expected_word_correction",
        "accepted_by_homophone",
        "accepted_by_phoneme_similarity",
        "accepted_by_gop",
        "accepted_by_asr_spelling_variant",
        "accepted_by_split_merge",
    }
    return {
        "expected_tokens": normalize_for_wer(expected_text).split(),
        "raw_tokens": normalize_for_wer(raw_transcript).split(),
        "selected_operations": [
            {
                "operation": item.get("operation"),
                "expected_chunk": item.get("expected_chunk"),
                "recognized_chunk": item.get("recognized_chunk"),
                "status": item.get("status"),
                "alignment_confidence": item.get("alignment_confidence"),
                "reason": item.get("dynamic_correction_reason"),
            }
            for item in alignment
            if item.get("expected_word") is not None
        ],
        "after_alignment_summary": {
            "total_expected_words": sum(1 for item in alignment if item.get("expected_word") is not None),
            "correct_words_exact": sum(1 for item in alignment if item.get("status") == "exact_correct"),
            "correct_words_dynamic_accepted": sum(1 for item in alignment if item.get("status") in {"accepted_by_dynamic_expected_word_correction", "accepted_by_asr_spelling_variant"}),
            "correct_words_split_merge_accepted": sum(1 for item in alignment if item.get("status") == "accepted_by_split_merge"),
            "correct_words_homophone_accepted": sum(1 for item in alignment if item.get("status") == "accepted_by_homophone"),
            "correct_words_gop_accepted": sum(1 for item in alignment if item.get("status") == "accepted_by_gop"),
            "partial_words": sum(1 for item in alignment if item.get("status") == "partial"),
            "incorrect_words": sum(1 for item in alignment if item.get("status") == "incorrect"),
            "missing_words": sum(1 for item in alignment if item.get("status") == "missing"),
            "inserted_words": sum(1 for item in alignment if item.get("status") == "inserted"),
            "wcpm_correct_statuses": sorted(accepted_statuses),
        },
    }


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
        "dynamic_suspicious_fragment": bool(meta.get("dynamic_suspicious_fragment", False)),
        "dynamic_fragment_reasons": list(meta.get("dynamic_fragment_reasons", []) or []),
        "dynamic_phoneme_coverage": meta.get("dynamic_phoneme_coverage"),
        "asr_spelling_variant_enabled": bool(meta.get("asr_spelling_variant_enabled", True)),
        "asr_spelling_variant_applied": bool(meta.get("asr_spelling_variant_applied", False)),
        "asr_spelling_variant_strategy": str(meta.get("asr_spelling_variant_strategy", "dynamic_asr_spelling_variant")),
        "asr_spelling_variant_sub_strategy": str(meta.get("asr_spelling_variant_sub_strategy", "")),
        "asr_spelling_variant_confidence": meta.get("asr_spelling_variant_confidence"),
        "asr_spelling_variant_threshold": meta.get("asr_spelling_variant_threshold"),
        "consonant_skeleton_similarity": meta.get("consonant_skeleton_similarity"),
        "vowel_tolerant_similarity": meta.get("vowel_tolerant_similarity"),
        "expected_phoneme_coverage": meta.get("expected_phoneme_coverage"),
        "variant_edit_similarity": meta.get("variant_edit_similarity"),
        "variant_reason": str(meta.get("variant_reason", "")),
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
        "dynamic_suspicious_fragment": result.suspicious_fragment,
        "dynamic_fragment_reasons": result.fragment_reasons or [],
        "dynamic_phoneme_coverage": result.phoneme_coverage,
        "asr_spelling_variant_enabled": result.asr_spelling_variant_enabled,
        "asr_spelling_variant_applied": result.asr_spelling_variant_applied,
        "asr_spelling_variant_strategy": result.asr_spelling_variant_strategy,
        "asr_spelling_variant_sub_strategy": result.asr_spelling_variant_sub_strategy,
        "asr_spelling_variant_confidence": result.asr_spelling_variant_confidence,
        "asr_spelling_variant_threshold": result.asr_spelling_variant_threshold,
        "consonant_skeleton_similarity": result.consonant_skeleton_similarity,
        "vowel_tolerant_similarity": result.vowel_tolerant_similarity,
        "expected_phoneme_coverage": result.expected_phoneme_coverage,
        "variant_edit_similarity": result.variant_edit_similarity,
        "variant_reason": result.variant_reason,
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
    if known_confusion >= 0.90 and context_score >= 0.95 and (spelling >= 0.50 or (phoneme_score or 0.0) >= 0.70 or (gop_score or 0.0) >= 0.75):
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


def _variant_weighted_score(
    *,
    prompt_type: str,
    consonant_skeleton_similarity: float,
    vowel_tolerant_similarity: float,
    gop_score: float | None,
    expected_phoneme_coverage: float | None,
    edit_similarity: float,
    context_score: float,
) -> float:
    if prompt_type == "letter":
        weights = {"known": 0.30, "phoneme": 0.25, "gop": 0.20, "edit": 0.15, "context": 0.10}
        values = {
            "known": consonant_skeleton_similarity,
            "phoneme": expected_phoneme_coverage,
            "gop": gop_score,
            "edit": edit_similarity,
            "context": context_score,
        }
    else:
        weights = {"skeleton": 0.25, "vowel": 0.20, "gop": 0.20, "coverage": 0.15, "edit": 0.10, "context": 0.10}
        values = {
            "skeleton": consonant_skeleton_similarity,
            "vowel": vowel_tolerant_similarity,
            "gop": gop_score,
            "coverage": expected_phoneme_coverage,
            "edit": edit_similarity,
            "context": context_score,
        }
    total_weight = sum(weight for key, weight in weights.items() if values.get(key) is not None)
    if total_weight <= 0:
        return 0.0
    score = sum(weights[key] * float(values[key]) for key in weights if values.get(key) is not None) / total_weight
    return round(max(0.0, min(1.0, score)), 6)


def _variant_result(
    accepted: bool,
    corrected_text: str | None,
    display_text: str,
    sub_strategy: str,
    confidence: float,
    threshold: float,
    consonant_skeleton_similarity: float,
    vowel_tolerant_similarity: float,
    edit_similarity: float,
    phoneme_similarity_score: float | None,
    gop_score: float | None,
    expected_phoneme_coverage: float | None,
    context_score: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "is_variant": bool(accepted),
        "accepted": bool(accepted),
        "corrected_text": corrected_text,
        "display_text": display_text,
        "strategy": "dynamic_asr_spelling_variant",
        "sub_strategy": sub_strategy,
        "confidence": round(confidence, 6),
        "threshold": round(threshold, 6),
        "consonant_skeleton_similarity": round(consonant_skeleton_similarity, 6),
        "vowel_tolerant_similarity": round(vowel_tolerant_similarity, 6),
        "edit_similarity": round(edit_similarity, 6),
        "phoneme_similarity": None if phoneme_similarity_score is None else round(phoneme_similarity_score, 6),
        "gop_score": None if gop_score is None else round(gop_score, 6),
        "expected_phoneme_coverage": None if expected_phoneme_coverage is None else round(expected_phoneme_coverage, 6),
        "context_score": round(context_score, 6),
        "reason": reason,
        "asr_spelling_variant_enabled": True,
        "asr_spelling_variant_applied": bool(accepted),
        "asr_spelling_variant_strategy": "dynamic_asr_spelling_variant",
        "asr_spelling_variant_sub_strategy": sub_strategy,
        "asr_spelling_variant_confidence": round(confidence, 6),
        "asr_spelling_variant_threshold": round(threshold, 6),
        "variant_edit_similarity": round(edit_similarity, 6),
        "variant_reason": reason,
    }


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
    suspicious_fragment: bool = False,
    fragment_reasons: list[str] | None = None,
    phoneme_coverage: float | None = None,
    strategy: str = "dynamic_expected_word_correction",
    variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant = variant or {}
    return DynamicCorrectionResult(
        accepted=accepted,
        corrected_text=corrected,
        display_text=display,
        strategy=strategy,
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
        suspicious_fragment=suspicious_fragment,
        fragment_reasons=fragment_reasons or [],
        phoneme_coverage=phoneme_coverage,
        asr_spelling_variant_enabled=bool(variant.get("asr_spelling_variant_enabled", True)),
        asr_spelling_variant_applied=bool(variant.get("asr_spelling_variant_applied", False)),
        asr_spelling_variant_strategy=str(variant.get("asr_spelling_variant_strategy", "dynamic_asr_spelling_variant")),
        asr_spelling_variant_sub_strategy=str(variant.get("asr_spelling_variant_sub_strategy", "")),
        asr_spelling_variant_confidence=variant.get("asr_spelling_variant_confidence"),
        asr_spelling_variant_threshold=variant.get("asr_spelling_variant_threshold"),
        consonant_skeleton_similarity=variant.get("consonant_skeleton_similarity"),
        vowel_tolerant_similarity=variant.get("vowel_tolerant_similarity"),
        expected_phoneme_coverage=variant.get("expected_phoneme_coverage"),
        variant_edit_similarity=variant.get("variant_edit_similarity"),
        variant_reason=str(variant.get("variant_reason", "")),
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
        "fragment_detection_enabled": _as_bool(merged.get("fragment_detection_enabled", True)),
        "reject_consonant_fragment_by_default": _as_bool(merged.get("reject_consonant_fragment_by_default", True)),
        "min_raw_length_ratio_for_words": float(merged.get("min_raw_length_ratio_for_words", 0.50)),
        "min_phoneme_coverage_for_words": float(merged.get("min_phoneme_coverage_for_words", 0.65)),
        "fragment_gop_accept_threshold": float(merged.get("fragment_gop_accept_threshold", 0.88)),
        "fragment_phoneme_accept_threshold": float(merged.get("fragment_phoneme_accept_threshold", 0.82)),
        "fragment_retry_on_bad_audio": _as_bool(merged.get("fragment_retry_on_bad_audio", True)),
        "fragment_allow_accept_with_strong_gop": _as_bool(merged.get("fragment_allow_accept_with_strong_gop", True)),
        "asr_spelling_variant_enabled": _as_bool(merged.get("asr_spelling_variant_enabled", True)),
        "asr_variant_word_accept_threshold": float(merged.get("asr_variant_word_accept_threshold", 0.78)),
        "asr_variant_rhyme_accept_threshold": float(merged.get("asr_variant_rhyme_accept_threshold", 0.78)),
        "asr_variant_letter_accept_threshold": float(merged.get("asr_variant_letter_accept_threshold", 0.72)),
        "asr_variant_sentence_word_accept_threshold": float(merged.get("asr_variant_sentence_word_accept_threshold", 0.82)),
        "asr_variant_passage_word_accept_threshold": float(merged.get("asr_variant_passage_word_accept_threshold", 0.84)),
        "asr_variant_min_consonant_skeleton": float(merged.get("asr_variant_min_consonant_skeleton", 0.80)),
        "asr_variant_min_vowel_tolerant_score": float(merged.get("asr_variant_min_vowel_tolerant_score", 0.78)),
        "asr_variant_min_gop_for_risky_accept": float(merged.get("asr_variant_min_gop_for_risky_accept", 0.82)),
        "asr_variant_min_phoneme_coverage": float(merged.get("asr_variant_min_phoneme_coverage", 0.70)),
        "asr_variant_short_word_strict": _as_bool(merged.get("asr_variant_short_word_strict", True)),
        "asr_variant_debug": _as_bool(merged.get("asr_variant_debug", True)),
        "dynamic_alignment_repair_enabled": _as_bool(merged.get("dynamic_alignment_repair_enabled", True)),
        "dynamic_alignment_allow_split_merge": _as_bool(merged.get("dynamic_alignment_allow_split_merge", True)),
        "dynamic_alignment_max_expected_chunk": int(merged.get("dynamic_alignment_max_expected_chunk", 3)),
        "dynamic_alignment_max_raw_chunk": int(merged.get("dynamic_alignment_max_raw_chunk", 3)),
        "dynamic_alignment_accept_threshold": float(merged.get("dynamic_alignment_accept_threshold", 0.82)),
        "dynamic_alignment_partial_threshold": float(merged.get("dynamic_alignment_partial_threshold", 0.65)),
        "dynamic_alignment_split_merge_threshold": float(merged.get("dynamic_alignment_split_merge_threshold", 0.84)),
        "dynamic_alignment_homophone_threshold": float(merged.get("dynamic_alignment_homophone_threshold", 0.96)),
        "dynamic_alignment_gop_accept_threshold": float(merged.get("dynamic_alignment_gop_accept_threshold", 0.75)),
        "dynamic_alignment_short_function_word_strict": _as_bool(merged.get("dynamic_alignment_short_function_word_strict", True)),
        "dynamic_alignment_debug": _as_bool(merged.get("dynamic_alignment_debug", True)),
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


def _variant_threshold_for_prompt(prompt_type: str, config: dict[str, Any]) -> float:
    if prompt_type == "letter":
        return float(config["asr_variant_letter_accept_threshold"])
    if prompt_type in {"rhyme", "rhyming_word"}:
        return float(config["asr_variant_rhyme_accept_threshold"])
    if prompt_type in {"sentence", "final_sentence"}:
        return float(config["asr_variant_sentence_word_accept_threshold"])
    if prompt_type in {"paragraph", "passage", "reading_passage", "final_passage"}:
        return float(config["asr_variant_passage_word_accept_threshold"])
    return float(config["asr_variant_word_accept_threshold"])


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
    return phoneme_sequence_similarity(expected_phonemes, raw_phonemes)


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


def _consonant_skeleton_similarity(expected: str, raw: str) -> float:
    left = _consonant_skeleton(expected)
    right = _consonant_skeleton(raw)
    if not left and not right:
        return _spelling_similarity(expected, raw)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if _is_subsequence(right, left) or _is_subsequence(left, right):
        length_ratio = min(len(left), len(right)) / max(len(left), len(right))
        return round(max(SequenceMatcher(a=left, b=right).ratio(), length_ratio), 6)
    return _spelling_similarity(left, right)


def _consonant_skeleton(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalpha() and char not in "aeiouy")


def _vowel_tolerant_similarity(expected: str, raw: str) -> float:
    left = "".join(char for char in expected.lower() if char.isalpha())
    right = "".join(char for char in raw.lower() if char.isalpha())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    n = len(left)
    m = len(right)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = _vowel_tolerant_substitution_cost(left[i - 1], right[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + substitution,
            )
    distance = dp[n][m]
    return round(max(0.0, min(1.0, 1.0 - (distance / max(n, m)))), 6)


def _vowel_tolerant_substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    left_vowel = left in "aeiouy"
    right_vowel = right in "aeiouy"
    if left_vowel and right_vowel:
        return 0.25
    return 1.0


def _short_word_fragment(
    *,
    expected: str,
    raw: str,
    prompt_type: str,
    phoneme_score: float | None,
    audio_quality: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    bad_audio = _bad_audio_for_fragment(audio_quality)

    if not bool(config["fragment_detection_enabled"]) or prompt_type == "letter":
        return {"suspicious_fragment": False, "fragment_reasons": reasons, "bad_audio": bad_audio, "phoneme_coverage": phoneme_score}

    expected_compact = expected.replace(" ", "")
    raw_compact = raw.replace(" ", "")
    isolated_word = prompt_type in {"word", "rhyme", "rhyming_word"}
    expected_has_vowel = _has_vowel_like_character(expected_compact)
    raw_has_vowel = _has_vowel_like_character(raw_compact)
    phoneme_coverage = phoneme_score

    if not raw_compact:
        reasons.append("blank_raw_transcript")
    if len(raw_compact) <= 1 and len(expected_compact) > 1:
        reasons.append("raw_too_short_for_non_letter")
    if expected_has_vowel and not raw_has_vowel:
        reasons.append("raw_missing_vowel_like_character")
    if isolated_word and expected_compact:
        ratio = len(raw_compact) / max(1, len(expected_compact))
        if ratio < float(config["min_raw_length_ratio_for_words"]):
            reasons.append("raw_shorter_than_expected_ratio")
    if bool(config["reject_consonant_fragment_by_default"]) and _looks_like_consonant_skeleton(expected_compact, raw_compact):
        reasons.append("raw_looks_like_consonant_skeleton")
    if reasons and phoneme_coverage is not None and phoneme_coverage < float(config["min_phoneme_coverage_for_words"]):
        reasons.append("incomplete_phoneme_coverage")
    if bad_audio:
        reasons.append("bad_audio_quality")

    return {
        "suspicious_fragment": bool(reasons),
        "fragment_reasons": reasons,
        "bad_audio": bad_audio,
        "phoneme_coverage": phoneme_coverage,
    }


def _has_vowel_like_character(value: str) -> bool:
    return any(char in "aeiouy" for char in value.lower())


def _looks_like_consonant_skeleton(expected: str, raw: str) -> bool:
    if not expected or not raw or len(raw) >= len(expected):
        return False
    skeleton = "".join(char for char in expected.lower() if char.isalpha() and char not in "aeiouy")
    compact_raw = "".join(char for char in raw.lower() if char.isalpha())
    if len(compact_raw) < 2 or not skeleton:
        return False
    return compact_raw == skeleton or _is_subsequence(compact_raw, skeleton)


def _is_subsequence(needle: str, haystack: str) -> bool:
    position = 0
    for char in haystack:
        if position < len(needle) and needle[position] == char:
            position += 1
    return position == len(needle)


def _bad_audio_for_fragment(audio_quality: dict[str, Any] | None) -> bool:
    if not audio_quality:
        return False
    flags = audio_quality.get("quality_flags") if isinstance(audio_quality.get("quality_flags"), dict) else {}
    if bool(
        audio_quality.get("mostly_silent")
        or audio_quality.get("too_short")
        or audio_quality.get("severe_clipping")
        or flags.get("mostly_silent")
        or flags.get("too_short")
        or flags.get("clipped")
    ):
        return True
    quality = str(audio_quality.get("quality") or audio_quality.get("status") or "").lower()
    if quality in {"bad", "unusable", "mostly_silent", "too_short"}:
        return True
    speech_duration = audio_quality.get("speech_duration_seconds")
    try:
        return bool(audio_quality.get("audio_valid", True)) and speech_duration is not None and float(speech_duration) < 0.35
    except (TypeError, ValueError):
        return False


def _critical_phoneme_contradicted(meta: dict[str, Any]) -> bool:
    return bool(meta.get("critical_pair_detected", False)) and meta.get("critical_phoneme_detected") is False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
