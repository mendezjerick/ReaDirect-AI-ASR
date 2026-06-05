from __future__ import annotations

import math
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.phoneme_comparison import phoneme_similarity
from readirect_asr.text.normalization import normalize_for_wer
from readirect_asr.text.transcript_normalizer import detect_prompt_type, generate_expected_phonemes


DEFAULT_GOP_CONFIG = {
    "enabled": True,
    "letter_threshold": 0.70,
    "word_threshold": 0.75,
    "rhyme_threshold": 0.75,
    "sentence_word_threshold": 0.70,
    "passage_word_threshold": 0.70,
    "min_alignment_quality": 0.25,
    "weak_threshold": 0.55,
    "acceptable_threshold": 0.75,
    "min_audio_quality_required": True,
    "skip_on_retry_required": True,
    "skip_on_uncertain_audio": True,
    "debug": False,
}

SHORT_PROMPT_TYPES = {"letter", "word", "rhyme", "rhyming_word"}
LONG_PROMPT_TYPES = {"sentence", "paragraph", "passage", "final_sentence", "reading_passage"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SPECIAL_TOKENS = {"", "<PAD>", "<S>", "</S>", "<UNK>", "[PAD]", "[UNK]", "|", "SIL", "SPN", "UNK", "PAD", "BLANK"}

LETTER_SOUND_PHONEMES = {
    "A": ["AE"],
    "B": ["B"],
    "C": ["K"],
    "D": ["D"],
    "E": ["EH"],
    "F": ["F"],
    "G": ["G"],
    "H": ["HH"],
    "I": ["IH"],
    "J": ["JH"],
    "K": ["K"],
    "L": ["L"],
    "M": ["M"],
    "N": ["N"],
    "O": ["AA"],
    "P": ["P"],
    "Q": ["K"],
    "R": ["R"],
    "S": ["S"],
    "T": ["T"],
    "U": ["AH"],
    "V": ["V"],
    "W": ["W"],
    "X": ["K", "S"],
    "Y": ["Y"],
    "Z": ["Z"],
}

LETTER_SOUND_HINTS = {
    "letter_sound",
    "sound_drill",
    "see_letter_say_sound",
    "match_sound_to_letter",
    "hear_and_repeat",
    "listen_and_say",
    "mastery_check",
}

ARPABET_TO_MODEL_LABELS = {
    "AA": ["ɑ", "ɑː", "a"],
    "AE": ["æ", "a"],
    "AH": ["ʌ", "ə", "ɐ"],
    "AO": ["ɔ", "ɔː", "o"],
    "AW": ["aʊ"],
    "AY": ["aɪ"],
    "EH": ["ɛ", "e"],
    "ER": ["ɚ", "ɜː", "ɜ", "ə"],
    "EY": ["eɪ", "e"],
    "IH": ["ɪ", "i"],
    "IY": ["i", "iː"],
    "OW": ["oʊ", "o"],
    "OY": ["ɔɪ", "oɪ"],
    "UH": ["ʊ", "u"],
    "UW": ["u", "uː"],
    "B": ["b"],
    "CH": ["tʃ", "tS"],
    "D": ["d"],
    "DH": ["ð"],
    "F": ["f"],
    "G": ["ɡ", "g"],
    "HH": ["h"],
    "JH": ["dʒ", "dZ"],
    "K": ["k"],
    "L": ["l"],
    "M": ["m"],
    "N": ["n"],
    "NG": ["ŋ"],
    "P": ["p"],
    "R": ["ɹ", "r"],
    "S": ["s"],
    "SH": ["ʃ"],
    "T": ["t"],
    "TH": ["θ"],
    "V": ["v"],
    "W": ["w"],
    "Y": ["j", "y"],
    "Z": ["z"],
    "ZH": ["ʒ"],
}


def compute_gop(
    audio_path_or_waveform: Any,
    expected_text: str,
    prompt_type: str,
    raw_transcript: str | None = None,
    sample_rate: int = 16000,
    phoneme_model: Any | None = None,
    phoneme_processor: Any | None = None,
    *,
    task_type: str | None = None,
    observed_phonemes: list[str] | None = None,
    acoustic_evidence: dict[str, Any] | None = None,
    cmudict_loader: CMUDictLoader | None = None,
    config: dict[str, Any] | None = None,
    audio_quality: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
) -> dict[str, Any]:
    del audio_path_or_waveform, sample_rate, phoneme_model, phoneme_processor
    active_config = {**DEFAULT_GOP_CONFIG, **(config or {})}
    detected_prompt_type = detect_prompt_type(expected_text, prompt_type=prompt_type)
    threshold = _threshold_for_prompt(detected_prompt_type, active_config)
    base = _base_payload(active_config, detected_prompt_type, threshold)

    if not bool(active_config.get("enabled", True)):
        return _unsupported(base, "GOP is disabled", enabled=False, decision="disabled")

    expected = str(expected_text or "").strip()
    if expected == "":
        return _unsupported(base, "expected_text is missing", decision="skipped_no_expected_text")

    if detected_prompt_type not in SHORT_PROMPT_TYPES | LONG_PROMPT_TYPES:
        return _unsupported(base, f"unsupported prompt type: {detected_prompt_type}", decision="skipped_unsupported_prompt_type")

    bad_audio_reason = _bad_audio_reason(audio_quality or {}, retry_required, uncertain, active_config)
    if bad_audio_reason:
        return _unsupported(base, bad_audio_reason, decision="skipped_bad_audio")

    try:
        expected_phonemes, expected_source, expected_variants = canonical_expected_phonemes(
            expected,
            prompt_type=detected_prompt_type,
            task_type=task_type,
            cmudict_loader=cmudict_loader,
        )
        if not expected_phonemes:
            return _unsupported(base, "expected phonemes are unavailable", decision="not_available")

        evidence = acoustic_evidence or {}
        if not evidence.get("available"):
            return {
                **_unsupported(base, str(evidence.get("error") or "phoneme frame probabilities are unavailable"), decision="not_available"),
                "gop_expected_phonemes": expected_phonemes,
                "canonical_phonemes": expected_phonemes,
                "canonical_expected_phonemes": expected_phonemes,
                "expected_phoneme_source": expected_source,
                "expected_phoneme_variants": expected_variants,
            }
        model_expected_phonemes, missing_model_labels = _map_expected_to_model_labels(expected_phonemes, evidence.get("vocabulary") or {})
        if missing_model_labels:
            return {
                **_unsupported(base, f"phoneme token not found in model vocabulary: {', '.join(missing_model_labels)}", decision="alignment_failed"),
                "gop_expected_phonemes": expected_phonemes,
                "canonical_phonemes": expected_phonemes,
                "canonical_expected_phonemes": expected_phonemes,
                "gop_model_version": evidence.get("model_version") or base["gop_model_version"],
                "gop_model_path": evidence.get("model_path"),
                "alignment_quality": "failed",
            }

        alignment = ctc_forced_align(
            expected_phonemes=model_expected_phonemes,
            frame_log_probs=evidence.get("log_probs"),
            vocabulary=evidence.get("vocabulary") or {},
            blank_token_id=evidence.get("blank_token_id"),
        )
        if not alignment["ok"]:
            return {
                **_unsupported(base, str(alignment["error"]), decision="alignment_failed"),
                "gop_expected_phonemes": expected_phonemes,
                "canonical_phonemes": expected_phonemes,
                "canonical_expected_phonemes": expected_phonemes,
                "gop_model_version": evidence.get("model_version") or base["gop_model_version"],
                "gop_model_path": evidence.get("model_path"),
                "alignment_quality": "failed",
            }

        phoneme_scores = score_aligned_phonemes(
            alignment["segments"],
            evidence.get("log_probs"),
            alignment["phone_ids"],
            evidence.get("vocabulary") or {},
            evidence.get("blank_token_id"),
            float(evidence.get("duration_seconds") or 0.0),
            weak_threshold=float(active_config.get("weak_threshold", 0.55)),
            acceptable_threshold=float(active_config.get("acceptable_threshold", 0.75)),
        )
        if not phoneme_scores:
            return {
                **_unsupported(base, "alignment produced no phoneme score ranges", decision="alignment_failed"),
                "gop_expected_phonemes": expected_phonemes,
                "canonical_phonemes": expected_phonemes,
                "canonical_expected_phonemes": expected_phonemes,
                "alignment_quality": "failed",
            }
        for index, item in enumerate(phoneme_scores):
            if index < len(expected_phonemes):
                item["model_phoneme"] = item.get("phoneme")
                item["phoneme"] = expected_phonemes[index]
                item["expected_phoneme"] = expected_phonemes[index]

        overall = _mean([float(item["score"]) for item in phoneme_scores])
        lowest = min(phoneme_scores, key=lambda item: float(item["score"]))
        min_alignment_quality = float(active_config.get("min_alignment_quality", 0.25))
        alignment_quality = "usable" if overall >= min_alignment_quality else "low_confidence"
        observed = _normalize_phonemes(observed_phonemes or [])
        decoded = _normalize_phonemes(evidence.get("decoded_phonemes") or observed)
        sequence_similarity = phoneme_similarity(expected_phonemes, decoded) if decoded else 0.0
        transcript_support = _transcript_support(expected, raw_transcript or "")
        decision = "accepted_by_pronunciation_evidence" if overall >= threshold else "rejected_low_gop"

        return {
            **base,
            "gop_available": True,
            "gop_supported": True,
            "gop_score": round(overall, 6),
            "overall_gop_score": round(overall, 6),
            "gop_confidence": round(_mean([overall, sequence_similarity, transcript_support]), 6),
            "acoustic_confidence": round(overall, 6),
            "gop_decision": decision,
            "gop_expected_phonemes": expected_phonemes,
            "canonical_phonemes": expected_phonemes,
            "canonical_expected_phonemes": expected_phonemes,
            "gop_observed_phonemes": decoded,
            "decoded_phonemes": decoded,
            "decoded_acoustic_phonemes": decoded,
            "gop_phoneme_scores": phoneme_scores,
            "phoneme_scores": phoneme_scores,
            "gop_word_scores": _word_scores(expected, detected_prompt_type, overall, threshold),
            "mispronounced_phonemes": [str(item["phoneme"]) for item in phoneme_scores if item["status"] == "weak"],
            "weak_words": [],
            "lowest_phoneme": lowest["phoneme"],
            "weak_phoneme": lowest["phoneme"],
            "lowest_phoneme_score": lowest["score"],
            "weak_phoneme_score": lowest["score"],
            "nearest_confusion": lowest.get("nearest_competitor"),
            "alignment_quality": alignment_quality,
            "gop_model_version": evidence.get("model_version") or base["gop_model_version"],
            "gop_model_path": evidence.get("model_path"),
            "gop_frame_count": evidence.get("frame_count"),
            "gop_duration_seconds": evidence.get("duration_seconds"),
            "gop_fallback_used": False,
            "gop_error": None,
            "debug_components": {
                "expected_phoneme_source": expected_source,
                "expected_phoneme_variants": expected_variants,
                "phoneme_sequence_similarity": round(sequence_similarity, 6),
                "transcript_support_score": round(transcript_support, 6),
                "alignment_path_log_score": alignment.get("path_log_score"),
            },
        }
    except Exception as exc:
        return _unsupported(base, str(exc), decision="error")


def canonical_expected_phonemes(
    expected_text: str,
    *,
    prompt_type: str,
    task_type: str | None = None,
    cmudict_loader: CMUDictLoader | None = None,
) -> tuple[list[str], str, list[list[str]]]:
    expected = str(expected_text or "").strip()
    task = str(task_type or "").lower()
    normalized_letter = expected.upper() if len(expected) == 1 and expected.isalpha() else ""
    if normalized_letter and (prompt_type == "letter_sound" or "letter_sound" in task or task in LETTER_SOUND_HINTS or "sound" in task):
        phones = LETTER_SOUND_PHONEMES.get(normalized_letter, [])
        return phones, "letter_sound_map", [phones] if phones else []

    phones, source, variants = generate_expected_phonemes(expected, cmudict_loader)
    if not phones:
        phones = _fallback_expected_phonemes(expected)
        source = "simple_fallback" if phones else source
        variants = [phones] if phones else variants
    return _normalize_phonemes(phones), source, [_normalize_phonemes(variant) for variant in variants if variant]


def _map_expected_to_model_labels(expected_phonemes: list[str], vocabulary: dict[Any, Any]) -> tuple[list[str], list[str]]:
    vocab_labels = {_normalize_model_token(label) for label in vocabulary.values()}
    mapped: list[str] = []
    missing: list[str] = []

    for phone in expected_phonemes:
        normalized = str(phone).strip()
        base = "".join(ch for ch in normalized.upper() if not ch.isdigit())
        candidates = [
            normalized,
            normalized.upper(),
            normalized.lower(),
            base,
            base.lower(),
            *ARPABET_TO_MODEL_LABELS.get(base, []),
        ]
        match = next((_normalize_model_token(candidate) for candidate in candidates if _normalize_model_token(candidate) in vocab_labels), None)
        if match is None:
            missing.append(normalized)
            continue
        mapped.append(match)

    return mapped, missing


def ctc_forced_align(
    *,
    expected_phonemes: list[str],
    frame_log_probs: Any,
    vocabulary: dict[int, str] | dict[str, str],
    blank_token_id: int | None,
) -> dict[str, Any]:
    log_probs = _as_frame_matrix(frame_log_probs)
    if not log_probs:
        return {"ok": False, "error": "frame log probabilities are empty"}
    if not expected_phonemes:
        return {"ok": False, "error": "expected phoneme sequence is empty"}

    id_to_phone = {int(index): _normalize_model_token(label) for index, label in vocabulary.items()}
    phone_to_ids: dict[str, list[int]] = {}
    for index, phone in id_to_phone.items():
        if phone and phone not in SPECIAL_TOKENS:
            phone_to_ids.setdefault(phone, []).append(index)

    phone_ids: list[int] = []
    missing: list[str] = []
    for phone in [_normalize_model_token(item) for item in expected_phonemes]:
        ids = phone_to_ids.get(phone)
        if not ids:
            missing.append(phone)
            continue
        phone_ids.append(ids[0])
    if missing:
        return {"ok": False, "error": "phoneme token not found in model vocabulary: "+", ".join(missing)}
    if blank_token_id is None:
        blank_token_id = _infer_blank_id(id_to_phone)
    if blank_token_id is None:
        return {"ok": False, "error": "blank token id is unavailable"}

    frames = len(log_probs)
    labels: list[int] = [blank_token_id]
    label_to_phone_index: list[int | None] = [None]
    for index, phone_id in enumerate(phone_ids):
        labels.extend([phone_id, blank_token_id])
        label_to_phone_index.extend([index, None])
    states = len(labels)
    neg_inf = -1.0e30
    dp = [[neg_inf] * states for _ in range(frames)]
    back = [[0] * states for _ in range(frames)]
    dp[0][0] = _lp(log_probs, 0, labels[0])
    if states > 1:
        dp[0][1] = _lp(log_probs, 0, labels[1])
        back[0][1] = 0

    for frame in range(1, frames):
        for state in range(states):
            candidates = [(dp[frame - 1][state], state)]
            if state - 1 >= 0:
                candidates.append((dp[frame - 1][state - 1], state - 1))
            if state - 2 >= 0 and labels[state] != blank_token_id and labels[state] != labels[state - 2]:
                candidates.append((dp[frame - 1][state - 2], state - 2))
            best_score, best_state = max(candidates, key=lambda item: item[0])
            dp[frame][state] = best_score + _lp(log_probs, frame, labels[state])
            back[frame][state] = best_state

    final_candidates = [(dp[frames - 1][states - 1], states - 1)]
    if states > 1:
        final_candidates.append((dp[frames - 1][states - 2], states - 2))
    best_final_score, state = max(final_candidates, key=lambda item: item[0])
    if best_final_score <= neg_inf / 2:
        return {"ok": False, "error": "CTC alignment did not find a valid path"}

    state_path = [0] * frames
    for frame in range(frames - 1, -1, -1):
        state_path[frame] = state
        state = back[frame][state] if frame > 0 else state

    frame_groups: list[list[int]] = [[] for _ in phone_ids]
    for frame, aligned_state in enumerate(state_path):
        phone_index = label_to_phone_index[aligned_state]
        if phone_index is not None:
            frame_groups[phone_index].append(frame)

    segments: list[dict[str, Any]] = []
    for index, frames_for_phone in enumerate(frame_groups):
        if not frames_for_phone:
            return {"ok": False, "error": f"no aligned frames for phoneme {expected_phonemes[index]}"}
        segments.append({
            "phoneme": _normalize_model_token(expected_phonemes[index]),
            "phone_id": phone_ids[index],
            "start_frame": min(frames_for_phone),
            "end_frame": max(frames_for_phone) + 1,
            "frames": frames_for_phone,
        })

    return {
        "ok": True,
        "segments": segments,
        "phone_ids": phone_ids,
        "path_log_score": round(best_final_score / max(1, frames), 6),
    }


def score_aligned_phonemes(
    segments: list[dict[str, Any]],
    frame_log_probs: Any,
    phone_ids: list[int],
    vocabulary: dict[int, str] | dict[str, str],
    blank_token_id: int | None,
    duration_seconds: float,
    *,
    weak_threshold: float,
    acceptable_threshold: float,
) -> list[dict[str, Any]]:
    log_probs = _as_frame_matrix(frame_log_probs)
    id_to_phone = {int(index): _normalize_model_token(label) for index, label in vocabulary.items()}
    excluded = set(phone_ids)
    if blank_token_id is not None:
        excluded.add(int(blank_token_id))
    candidate_ids = [
        index for index, phone in id_to_phone.items()
        if index not in excluded and phone and phone not in SPECIAL_TOKENS
    ]
    frame_count = max(1, len(log_probs))
    ms_per_frame = (duration_seconds * 1000.0 / frame_count) if duration_seconds > 0 else 0.0
    scores: list[dict[str, Any]] = []

    for segment in segments:
        frames = list(segment["frames"])
        expected_id = int(segment["phone_id"])
        expected_values = [_lp(log_probs, frame, expected_id) for frame in frames]
        competitor_avgs = {
            index: _mean([_lp(log_probs, frame, index) for frame in frames])
            for index in candidate_ids
        }
        competitor_id, competitor_log_score = (None, None)
        if competitor_avgs:
            competitor_id, competitor_log_score = max(competitor_avgs.items(), key=lambda item: item[1])
        expected_log_score = _mean(expected_values)
        margin = expected_log_score - float(competitor_log_score if competitor_log_score is not None else expected_log_score)
        normalized = _normalize_margin(margin)
        scores.append({
            "phoneme": segment["phoneme"],
            "expected_phoneme": segment["phoneme"],
            "start_ms": int(round(segment["start_frame"] * ms_per_frame)),
            "end_ms": int(round(segment["end_frame"] * ms_per_frame)),
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "score": round(normalized, 6),
            "raw_gop_margin": round(margin, 6),
            "status": _phoneme_status(normalized, weak_threshold, acceptable_threshold),
            "nearest_competitor": id_to_phone.get(int(competitor_id)) if competitor_id is not None else None,
            "competitor_score": round(_logprob_to_prob(float(competitor_log_score)), 6) if competitor_log_score is not None else None,
            "expected_log_probability": round(expected_log_score, 6),
            "competitor_log_probability": round(float(competitor_log_score), 6) if competitor_log_score is not None else None,
        })
    return scores


def apply_gop_to_transcript_meta(transcript_meta: dict[str, Any], gop: dict[str, Any]) -> dict[str, Any]:
    updated = dict(transcript_meta)
    updated.update(gop_response_fields(gop))
    debug_metadata = dict(updated.get("debug_metadata", {}) or {})
    debug_metadata["gop_evidence"] = gop_evidence_object(updated, gop)
    updated["debug_metadata"] = debug_metadata

    if gop.get("gop_decision") != "accepted_by_pronunciation_evidence":
        return updated

    prompt_type = str(updated.get("prompt_type") or gop.get("gop_prompt_type") or "")
    if prompt_type not in SHORT_PROMPT_TYPES:
        return updated

    expected = str(updated.get("expected_text") or "").strip()
    if not expected:
        return updated

    existing_support = bool(updated.get("accepted")) or float(updated.get("phonetic_similarity_score", 0.0) or 0.0) >= float(updated.get("threshold_used", 1.0) or 1.0)
    if not existing_support:
        updated["normalization_reason"] = "Acoustic GOP supported expected phonemes, but expected-centric transcript evidence remained the final decision layer"
        return updated

    updated["corrected_transcript"] = expected
    updated["displayed_transcript"] = expected
    updated["accepted"] = True
    updated["normalization_applied"] = True
    updated["normalization_reason"] = "Acoustic GOP evidence supported existing expected-centric acceptance"
    updated["correction_strategy_used"] = "expected_centric_with_acoustic_gop"
    updated["accepted_by_phoneme_evidence"] = True
    updated["gop_correction_applied"] = True
    updated["corrected_wer"] = 0.0
    updated["corrected_cer"] = 0.0
    updated["composite_score"] = max(float(updated.get("composite_score", 0.0) or 0.0), float(gop.get("gop_score", 0.0) or 0.0))
    return updated


def gop_response_fields(gop: dict[str, Any] | None) -> dict[str, Any]:
    gop = gop or {}
    phoneme_scores = list(gop.get("phoneme_scores", gop.get("gop_phoneme_scores", [])) or [])
    observed = list(gop.get("decoded_acoustic_phonemes", gop.get("gop_observed_phonemes", [])) or [])
    expected = list(gop.get("canonical_expected_phonemes", gop.get("gop_expected_phonemes", [])) or [])
    return {
        "gop_enabled": bool(gop.get("gop_enabled", True)),
        "gop_available": bool(gop.get("gop_available", False)),
        "gop_supported": bool(gop.get("gop_supported", gop.get("gop_available", False))),
        "gop_score": gop.get("gop_score", gop.get("overall_gop_score")),
        "overall_gop_score": gop.get("overall_gop_score", gop.get("gop_score")),
        "gop_confidence": gop.get("gop_confidence", gop.get("acoustic_confidence")),
        "acoustic_confidence": gop.get("acoustic_confidence", gop.get("gop_confidence")),
        "gop_decision": str(gop.get("gop_decision", "not_available")),
        "gop_threshold": gop.get("gop_threshold"),
        "gop_prompt_type": str(gop.get("gop_prompt_type", "unknown")),
        "gop_expected_phonemes": expected,
        "canonical_phonemes": expected,
        "canonical_expected_phonemes": expected,
        "gop_observed_phonemes": observed,
        "decoded_phonemes": observed,
        "decoded_acoustic_phonemes": observed,
        "gop_phoneme_scores": phoneme_scores,
        "phoneme_scores": phoneme_scores,
        "gop_word_scores": list(gop.get("gop_word_scores", []) or []),
        "mispronounced_phonemes": list(gop.get("mispronounced_phonemes", []) or []),
        "weak_words": list(gop.get("weak_words", []) or []),
        "lowest_phoneme": gop.get("lowest_phoneme", gop.get("weak_phoneme")),
        "weak_phoneme": gop.get("weak_phoneme", gop.get("lowest_phoneme")),
        "lowest_phoneme_score": gop.get("lowest_phoneme_score", gop.get("weak_phoneme_score")),
        "weak_phoneme_score": gop.get("weak_phoneme_score", gop.get("lowest_phoneme_score")),
        "nearest_confusion": gop.get("nearest_confusion"),
        "alignment_quality": gop.get("alignment_quality"),
        "gop_model_version": gop.get("gop_model_version", "existing_wavtec_phoneme_model"),
        "gop_model_path": gop.get("gop_model_path"),
        "gop_frame_count": gop.get("gop_frame_count"),
        "gop_duration_seconds": gop.get("gop_duration_seconds"),
        "gop_fallback_used": bool(gop.get("gop_fallback_used", not bool(gop.get("gop_available", False)))),
        "gop_correction_applied": bool(gop.get("gop_correction_applied", False)),
        "gop_error": gop.get("gop_error"),
    }


def gop_evidence_object(transcript_meta: dict[str, Any], gop: dict[str, Any]) -> dict[str, Any]:
    fields = gop_response_fields(gop)
    return {
        "expected": transcript_meta.get("expected_text"),
        "transcript": transcript_meta.get("raw_transcript"),
        "expected_phonemes": list(transcript_meta.get("expected_phonemes", []) or []),
        "transcript_phonemes": list(transcript_meta.get("observed_phonemes", []) or []),
        "decoded_acoustic_phonemes": fields["decoded_acoustic_phonemes"],
        "gop": {
            "gop_supported": fields["gop_supported"],
            "overall_gop_score": fields["overall_gop_score"],
            "phoneme_scores": fields["phoneme_scores"],
            "lowest_phoneme": fields["lowest_phoneme"],
            "lowest_phoneme_score": fields["lowest_phoneme_score"],
            "nearest_confusion": fields["nearest_confusion"],
            "alignment_quality": fields["alignment_quality"],
            "gop_error": fields["gop_error"],
        },
    }


def _base_payload(config: dict[str, Any], prompt_type: str, threshold: float) -> dict[str, Any]:
    return {
        "gop_enabled": bool(config.get("enabled", True)),
        "gop_available": False,
        "gop_supported": False,
        "gop_score": None,
        "overall_gop_score": None,
        "gop_confidence": None,
        "acoustic_confidence": None,
        "gop_decision": "not_available",
        "gop_threshold": threshold,
        "gop_prompt_type": prompt_type,
        "gop_expected_phonemes": [],
        "canonical_phonemes": [],
        "canonical_expected_phonemes": [],
        "gop_observed_phonemes": [],
        "decoded_phonemes": [],
        "decoded_acoustic_phonemes": [],
        "gop_phoneme_scores": [],
        "phoneme_scores": [],
        "gop_word_scores": [],
        "mispronounced_phonemes": [],
        "weak_words": [],
        "lowest_phoneme": None,
        "lowest_phoneme_score": None,
        "nearest_confusion": None,
        "alignment_quality": None,
        "gop_model_version": "existing_wavtec_phoneme_model",
        "gop_model_path": None,
        "gop_frame_count": None,
        "gop_duration_seconds": None,
        "gop_fallback_used": True,
        "gop_correction_applied": False,
        "gop_error": None,
    }


def _unsupported(base: dict[str, Any], error: str, *, enabled: bool = True, decision: str = "not_available") -> dict[str, Any]:
    return {
        **base,
        "gop_enabled": enabled,
        "gop_available": False,
        "gop_supported": False,
        "gop_decision": decision,
        "alignment_quality": "failed" if enabled else None,
        "gop_fallback_used": True,
        "gop_error": error,
    }


def _threshold_for_prompt(prompt_type: str, config: dict[str, Any]) -> float:
    if prompt_type == "letter":
        return float(config.get("letter_threshold", 0.70))
    if prompt_type in {"rhyme", "rhyming_word"}:
        return float(config.get("rhyme_threshold", 0.75))
    if prompt_type in {"sentence", "final_sentence"}:
        return float(config.get("sentence_word_threshold", 0.70))
    if prompt_type in {"paragraph", "passage", "reading_passage"}:
        return float(config.get("passage_word_threshold", 0.70))
    return float(config.get("word_threshold", 0.75))


def _bad_audio_reason(audio_quality: dict[str, Any], retry_required: bool, uncertain: bool, config: dict[str, Any]) -> str:
    if retry_required and bool(config.get("skip_on_retry_required", True)):
        return "retry_required is true"
    if uncertain and bool(config.get("skip_on_uncertain_audio", True)):
        return "uncertain audio is true"
    if not bool(config.get("min_audio_quality_required", True)):
        return ""
    flags = dict(audio_quality.get("quality_flags", {}) or {})
    for key in ("mostly_silent", "no_speech_detected", "too_short", "clipped"):
        if flags.get(key):
            return f"audio quality flag {key} is true"
    if audio_quality.get("passed") is False or audio_quality.get("is_acceptable") is False:
        return "audio quality did not pass"
    return ""


def _normalize_phonemes(phonemes: list[str]) -> list[str]:
    normalized: list[str] = []
    for phoneme in phonemes:
        clean = _normalize_token(phoneme)
        if clean and clean not in SPECIAL_TOKENS:
            normalized.append(clean)
    return normalized


def _normalize_token(value: Any) -> str:
    return "".join(char for char in str(value or "").upper() if char.isalpha())


def _normalize_model_token(value: Any) -> str:
    return str(value or "").strip()


def _fallback_expected_phonemes(text: str) -> list[str]:
    word = normalize_for_wer(text).replace(" ", "")
    phones: list[str] = []
    index = 0
    digraphs = {
        "ch": ["CH"],
        "sh": ["SH"],
        "th": ["TH"],
        "ph": ["F"],
        "wh": ["W"],
        "ee": ["IY"],
        "ea": ["IY"],
        "oo": ["UW"],
        "ow": ["OW"],
        "ay": ["EY"],
        "ai": ["EY"],
        "oy": ["OY"],
        "oi": ["OY"],
    }
    letters = {
        "a": ["AE"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH"], "f": ["F"], "g": ["G"], "h": ["HH"],
        "i": ["IH"], "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "o": ["OW"], "p": ["P"],
        "q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"], "u": ["AH"], "v": ["V"], "w": ["W"], "x": ["K", "S"],
        "y": ["Y"], "z": ["Z"],
    }
    while index < len(word):
        pair = word[index : index + 2]
        if pair in digraphs:
            phones.extend(digraphs[pair])
            index += 2
            continue
        phones.extend(letters.get(word[index], []))
        index += 1
    return phones


def _as_frame_matrix(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    matrix: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, list):
            matrix.append([float(item) for item in row])
    return matrix


def _lp(log_probs: list[list[float]], frame: int, token_id: int) -> float:
    if frame < 0 or frame >= len(log_probs):
        return -1.0e30
    row = log_probs[frame]
    if token_id < 0 or token_id >= len(row):
        return -1.0e30
    return float(row[token_id])


def _infer_blank_id(id_to_phone: dict[int, str]) -> int | None:
    for index, phone in id_to_phone.items():
        if phone in SPECIAL_TOKENS:
            return index
    return None


def _normalize_margin(margin: float) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-float(margin)))))


def _logprob_to_prob(log_probability: float) -> float:
    try:
        return math.exp(float(log_probability))
    except OverflowError:
        return 0.0


def _phoneme_status(score: float, weak_threshold: float, acceptable_threshold: float) -> str:
    if score < weak_threshold:
        return "weak"
    if score < acceptable_threshold:
        return "acceptable"
    return "good"


def _word_scores(expected_text: str, prompt_type: str, score: float, threshold: float) -> list[dict[str, Any]]:
    words = normalize_for_wer(expected_text).split()
    if prompt_type == "letter" and not words:
        words = [expected_text.strip()]
    if not words:
        return []
    status = "good" if score >= max(0.85, threshold) else "acceptable" if score >= threshold else "weak"
    return [{"word": word, "score": round(score, 6), "status": status} for word in words]


def _transcript_support(expected_text: str, raw_transcript: str) -> float:
    expected = normalize_for_wer(expected_text)
    raw = normalize_for_wer(raw_transcript)
    if not expected and not raw:
        return 1.0
    if not expected or not raw:
        return 0.0
    return round(max(0.0, 1.0 - compute_cer(expected, raw)), 6)


def _mean(values: list[float]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0
