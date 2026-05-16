from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer
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
    "min_audio_quality_required": True,
    "skip_on_retry_required": True,
    "skip_on_uncertain_audio": True,
    "debug": False,
}

SHORT_PROMPT_TYPES = {"letter", "word", "rhyme", "rhyming_word"}
LONG_PROMPT_TYPES = {"sentence", "paragraph", "passage", "final_sentence", "reading_passage"}

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SIMILAR_PHONEME_GROUPS = [
    {"IY", "IH", "EY"},
    {"EH", "AE", "AH"},
    {"OW", "AO", "UH", "UW"},
    {"L", "R"},
    {"S", "Z"},
    {"T", "D"},
    {"P", "B"},
    {"K", "G"},
    {"F", "V"},
    {"TH", "DH"},
    {"CH", "JH", "SH", "ZH"},
]


def compute_gop(
    audio_path_or_waveform: Any,
    expected_text: str,
    prompt_type: str,
    raw_transcript: str | None = None,
    sample_rate: int = 16000,
    phoneme_model: Any | None = None,
    phoneme_processor: Any | None = None,
    *,
    observed_phonemes: list[str] | None = None,
    cmudict_loader: Any | None = None,
    config: dict[str, Any] | None = None,
    audio_quality: dict[str, Any] | None = None,
    retry_required: bool = False,
    uncertain: bool = False,
) -> dict[str, Any]:
    active_config = {**DEFAULT_GOP_CONFIG, **(config or {})}
    detected_prompt_type = detect_prompt_type(expected_text, prompt_type=prompt_type)
    threshold = _threshold_for_prompt(detected_prompt_type, active_config)
    base = _base_payload(active_config, detected_prompt_type, threshold)

    if not bool(active_config.get("enabled", True)):
        return {**base, "gop_enabled": False, "gop_available": False, "gop_decision": "not_available", "gop_error": "GOP is disabled"}

    expected = str(expected_text or "").strip()
    if expected == "":
        return {**base, "gop_available": False, "gop_decision": "skipped_no_expected_text", "gop_error": "expected_text is missing"}

    if detected_prompt_type not in SHORT_PROMPT_TYPES | LONG_PROMPT_TYPES:
        return {**base, "gop_available": False, "gop_decision": "skipped_unsupported_prompt_type", "gop_error": f"unsupported prompt type: {detected_prompt_type}"}

    bad_audio_reason = _bad_audio_reason(audio_quality or {}, retry_required, uncertain, active_config)
    if bad_audio_reason:
        return {**base, "gop_available": False, "gop_decision": "skipped_bad_audio", "gop_error": bad_audio_reason}

    try:
        expected_phonemes, expected_source, expected_variants = generate_expected_phonemes(expected, cmudict_loader)
        if not expected_phonemes:
            expected_phonemes = _fallback_expected_phonemes(expected)
            expected_source = "simple_fallback" if expected_phonemes else expected_source
            expected_variants = [expected_phonemes] if expected_phonemes else expected_variants
        observed = _normalize_phonemes(observed_phonemes or [])

        if not observed and phoneme_model is not None and phoneme_processor is not None and audio_path_or_waveform is not None:
            observed = _decode_observed_phonemes(audio_path_or_waveform, sample_rate, phoneme_model, phoneme_processor)

        if not expected_phonemes:
            return {**base, "gop_available": False, "gop_decision": "not_available", "gop_error": "expected phonemes are unavailable"}

        if not observed:
            return {
                **base,
                "gop_available": False,
                "gop_decision": "not_available",
                "gop_expected_phonemes": expected_phonemes,
                "gop_error": "observed phonemes are unavailable",
            }

        alignment = _align_phonemes(expected_phonemes, observed)
        phoneme_scores = [_phoneme_score_item(expected, actual) for expected, actual in alignment if expected]
        sequence_similarity = phoneme_similarity(expected_phonemes, observed)
        alignment_score = _mean([float(item["score"]) for item in phoneme_scores])
        acoustic_confidence = _mean([float(item["score"]) for item in phoneme_scores]) if phoneme_scores else None
        transcript_support = _transcript_support(expected, raw_transcript or "")
        gop_score = _weighted_score(
            {
                "phoneme_sequence_similarity": sequence_similarity,
                "phoneme_alignment_score": alignment_score,
                "acoustic_confidence_score": acoustic_confidence,
                "transcript_support_score": transcript_support,
            }
        )
        status = _status_for_score(gop_score, threshold)
        word_scores = _word_scores(expected, detected_prompt_type, gop_score, threshold)
        weak_words = [item["word"] for item in word_scores if item["status"] == "weak"]
        mispronounced = [item["expected_phoneme"] for item in phoneme_scores if item["status"] == "weak"]
        decision = "accepted_by_pronunciation_evidence" if status in {"good", "acceptable"} else "rejected_low_gop"

        return {
            **base,
            "gop_available": True,
            "gop_score": round(gop_score, 6),
            "gop_confidence": round(_mean([sequence_similarity, alignment_score, acoustic_confidence or alignment_score]), 6),
            "gop_decision": decision,
            "gop_expected_phonemes": expected_phonemes,
            "gop_observed_phonemes": observed,
            "gop_phoneme_scores": phoneme_scores,
            "gop_word_scores": word_scores,
            "mispronounced_phonemes": mispronounced,
            "weak_words": weak_words,
            "gop_error": None,
            "debug_components": {
                "expected_phoneme_source": expected_source,
                "expected_phoneme_variants": expected_variants,
                "phoneme_sequence_similarity": round(sequence_similarity, 6),
                "phoneme_alignment_score": round(alignment_score, 6),
                "acoustic_confidence_score": round(acoustic_confidence or 0.0, 6),
                "transcript_support_score": round(transcript_support, 6),
            },
        }
    except Exception as exc:
        return {**base, "gop_available": False, "gop_decision": "error", "gop_error": str(exc)}


def apply_gop_to_transcript_meta(transcript_meta: dict[str, Any], gop: dict[str, Any]) -> dict[str, Any]:
    updated = dict(transcript_meta)
    updated.update(gop_response_fields(gop))

    if gop.get("gop_decision") != "accepted_by_pronunciation_evidence":
        return updated

    prompt_type = str(updated.get("prompt_type") or gop.get("gop_prompt_type") or "")
    if prompt_type not in SHORT_PROMPT_TYPES:
        return updated

    expected = str(updated.get("expected_text") or "").strip()
    if not expected:
        return updated

    updated["corrected_transcript"] = expected
    updated["displayed_transcript"] = expected
    updated["accepted"] = True
    updated["normalization_applied"] = True
    updated["normalization_reason"] = "GOP pronunciation evidence strongly matched expected text"
    updated["correction_strategy_used"] = "gop_pronunciation_evidence"
    updated["accepted_by_phoneme_evidence"] = True
    updated["gop_correction_applied"] = True
    updated["corrected_wer"] = 0.0
    updated["corrected_cer"] = 0.0
    updated["composite_score"] = max(float(updated.get("composite_score", 0.0) or 0.0), float(gop.get("gop_score", 0.0) or 0.0))
    return updated


def gop_response_fields(gop: dict[str, Any] | None) -> dict[str, Any]:
    gop = gop or {}
    return {
        "gop_enabled": bool(gop.get("gop_enabled", True)),
        "gop_available": bool(gop.get("gop_available", False)),
        "gop_score": gop.get("gop_score"),
        "gop_confidence": gop.get("gop_confidence"),
        "gop_decision": str(gop.get("gop_decision", "not_available")),
        "gop_threshold": gop.get("gop_threshold"),
        "gop_prompt_type": str(gop.get("gop_prompt_type", "unknown")),
        "gop_expected_phonemes": list(gop.get("gop_expected_phonemes", []) or []),
        "gop_observed_phonemes": list(gop.get("gop_observed_phonemes", []) or []),
        "gop_phoneme_scores": list(gop.get("gop_phoneme_scores", []) or []),
        "gop_word_scores": list(gop.get("gop_word_scores", []) or []),
        "mispronounced_phonemes": list(gop.get("mispronounced_phonemes", []) or []),
        "weak_words": list(gop.get("weak_words", []) or []),
        "gop_correction_applied": bool(gop.get("gop_correction_applied", False)),
        "gop_error": gop.get("gop_error"),
    }


def _base_payload(config: dict[str, Any], prompt_type: str, threshold: float) -> dict[str, Any]:
    return {
        "gop_enabled": bool(config.get("enabled", True)),
        "gop_available": False,
        "gop_score": None,
        "gop_confidence": None,
        "gop_decision": "not_available",
        "gop_threshold": threshold,
        "gop_prompt_type": prompt_type,
        "gop_expected_phonemes": [],
        "gop_observed_phonemes": [],
        "gop_phoneme_scores": [],
        "gop_word_scores": [],
        "mispronounced_phonemes": [],
        "weak_words": [],
        "gop_correction_applied": False,
        "gop_error": None,
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


def _decode_observed_phonemes(audio_path_or_waveform: Any, sample_rate: int, phoneme_model: Any, phoneme_processor: Any) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for phoneme GOP decoding") from exc

    audio = audio_path_or_waveform
    if isinstance(audio_path_or_waveform, (str, bytes)):
        import librosa

        audio, sample_rate = librosa.load(str(audio_path_or_waveform), sr=sample_rate, mono=True)
    inputs = phoneme_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    device = next(phoneme_model.parameters()).device
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = phoneme_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded = phoneme_processor.batch_decode(predicted_ids)[0]
    return _normalize_phonemes(str(decoded).replace("|", " ").replace("/", " ").split())


def _normalize_phonemes(phonemes: list[str]) -> list[str]:
    normalized: list[str] = []
    for phoneme in phonemes:
        clean = "".join(char for char in str(phoneme or "").upper() if char.isalpha())
        if clean and clean not in {"SIL", "SPN", "UNK", "PAD", "BLANK"}:
            normalized.append(clean)
    return normalized


def _fallback_expected_phonemes(text: str) -> list[str]:
    word = normalize_for_wer(text).replace(" ", "")
    overrides = {
        "leo": ["L", "IY", "OW"],
        "layo": ["L", "EY", "OW"],
        "layoh": ["L", "EY", "OW"],
    }
    if word in overrides:
        return overrides[word]
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
        "a": ["AE"],
        "b": ["B"],
        "c": ["K"],
        "d": ["D"],
        "e": ["EH"],
        "f": ["F"],
        "g": ["G"],
        "h": ["HH"],
        "i": ["IH"],
        "j": ["JH"],
        "k": ["K"],
        "l": ["L"],
        "m": ["M"],
        "n": ["N"],
        "o": ["OW"],
        "p": ["P"],
        "q": ["K"],
        "r": ["R"],
        "s": ["S"],
        "t": ["T"],
        "u": ["AH"],
        "v": ["V"],
        "w": ["W"],
        "x": ["K", "S"],
        "y": ["Y"],
        "z": ["Z"],
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


def _align_phonemes(expected: list[str], observed: list[str]) -> list[tuple[str | None, str | None]]:
    rows = len(expected)
    cols = len(observed)
    dp = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    back: list[list[str]] = [[""] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = dp[i - 1][0] - 1.0
        back[i][0] = "up"
    for j in range(1, cols + 1):
        dp[0][j] = dp[0][j - 1] - 1.0
        back[0][j] = "left"
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            match = dp[i - 1][j - 1] + _pair_score(expected[i - 1], observed[j - 1])
            delete = dp[i - 1][j] - 1.0
            insert = dp[i][j - 1] - 1.0
            best = max(match, delete, insert)
            dp[i][j] = best
            back[i][j] = "diag" if best == match else "up" if best == delete else "left"
    aligned: list[tuple[str | None, str | None]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            aligned.append((expected[i - 1], observed[j - 1]))
            i -= 1
            j -= 1
        elif move == "up":
            aligned.append((expected[i - 1], None))
            i -= 1
        else:
            aligned.append((None, observed[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def _phoneme_score_item(expected: str, observed: str | None) -> dict[str, Any]:
    score = _pair_score(expected, observed)
    return {
        "expected_phoneme": expected,
        "observed_phoneme": observed,
        "score": round(score, 6),
        "status": _phoneme_status(score),
    }


def _pair_score(expected: str | None, observed: str | None) -> float:
    if not expected or not observed:
        return 0.0
    if expected == observed:
        return 0.94
    if _similar_phoneme(expected, observed):
        return 0.74 if expected in VOWELS or observed in VOWELS else 0.68
    if (expected in VOWELS) == (observed in VOWELS):
        return 0.45
    return 0.20


def _similar_phoneme(left: str, right: str) -> bool:
    return any(left in group and right in group for group in SIMILAR_PHONEME_GROUPS)


def _phoneme_status(score: float) -> str:
    if score >= 0.85:
        return "good"
    if score >= 0.65:
        return "acceptable"
    return "weak"


def _status_for_score(score: float, threshold: float) -> str:
    if score >= max(0.85, threshold):
        return "good"
    if score >= threshold:
        return "acceptable"
    return "weak"


def _word_scores(expected_text: str, prompt_type: str, score: float, threshold: float) -> list[dict[str, Any]]:
    words = normalize_for_wer(expected_text).split()
    if prompt_type == "letter" and not words:
        words = [expected_text.strip()]
    if not words:
        return []
    status = _status_for_score(score, threshold)
    return [{"word": word, "score": round(score, 6), "status": status} for word in words]


def _transcript_support(expected_text: str, raw_transcript: str) -> float:
    expected = normalize_for_wer(expected_text)
    raw = normalize_for_wer(raw_transcript)
    if not expected and not raw:
        return 1.0
    if not expected or not raw:
        return 0.0
    cer = compute_cer(expected, raw)
    edit_similarity = max(0.0, 1.0 - cer)
    ratio = SequenceMatcher(a=expected, b=raw).ratio()
    return round(max(edit_similarity, ratio), 6)


def _weighted_score(components: dict[str, float | None]) -> float:
    weights = {
        "phoneme_sequence_similarity": 0.50,
        "phoneme_alignment_score": 0.30,
        "acoustic_confidence_score": 0.15,
        "transcript_support_score": 0.05,
    }
    available = {key: value for key, value in components.items() if value is not None}
    weight_sum = sum(weights[key] for key in available)
    if weight_sum <= 0:
        return 0.0
    score = sum(float(value) * (weights[key] / weight_sum) for key, value in available.items())
    return max(0.0, min(1.0, score))


def _mean(values: list[float]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0
