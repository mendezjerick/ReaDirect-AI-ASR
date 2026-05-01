from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_wer
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.phoneme_comparison import phoneme_similarity
from readirect_asr.text.normalization import normalize_for_wer, normalize_transcript


DEFAULT_PHONETIC_ACCEPT_THRESHOLD = 0.88
DEFAULT_STRICT_WORD_THRESHOLD = 0.90
DEFAULT_SINGLE_LETTER_THRESHOLD = 0.85
DEFAULT_KNOWN_CONFUSION_THRESHOLD = 0.82
DEFAULT_PHONETIC_LATTICE_THRESHOLD = 0.85
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_LOW_CONFIDENCE_SIMILARITY_THRESHOLD = 0.95


LETTER_PRONUNCIATIONS = {
    "a": {"a", "ay", "aye"},
    "b": {"b", "bee", "be"},
    "c": {"c", "see", "sea"},
    "d": {"d", "dee"},
    "e": {"e", "ee"},
    "f": {"f", "ef"},
    "g": {"g", "gee", "jee"},
    "h": {"h", "aitch", "haitch"},
    "i": {"i", "eye"},
    "j": {"j", "jay"},
    "k": {"k", "kay"},
    "l": {"l", "el"},
    "m": {"m", "em"},
    "n": {"n", "en"},
    "o": {"o", "oh"},
    "p": {"p", "pee"},
    "q": {"q", "cue", "queue"},
    "r": {"r", "are"},
    "s": {"s", "ess"},
    "t": {"t", "tee", "tea"},
    "u": {"u", "you"},
    "v": {"v", "vee"},
    "w": {"w", "double you", "double u", "doubleu"},
    "x": {"x", "ex", "axe"},
    "y": {"y", "why"},
    "z": {"z", "zee", "zed"},
}


LETTER_LATTICE_VARIANTS = {
    "a": {"ai", "ey"},
    "b": {"bi", "by", "bii"},
    "c": {"si", "cy", "sii"},
    "d": {"de", "di", "dy", "dii"},
    "e": {"i", "ii", "yi"},
    "f": {"eff"},
    "g": {"gi", "gy", "gii"},
    "h": {"age", "each"},
    "i": {"ai", "ay"},
    "j": {"jai", "jey"},
    "k": {"kei", "key"},
    "l": {"ell"},
    "m": {"emm"},
    "n": {"enn"},
    "o": {"owe"},
    "p": {"pi", "py", "pii"},
    "q": {"ku", "kyoo"},
    "r": {"ar"},
    "s": {"es"},
    "t": {"ti", "ty", "tii"},
    "u": {"yu", "yoo"},
    "v": {"vi", "vy", "vii"},
    "w": {"double-u", "dubya"},
    "x": {"eks"},
    "y": {"wai", "wy"},
    "z": {"zi", "zii", "zih", "zy", "zey", "ze", "zhee"},
}


LETTER_HIGH_FRONT_VOWEL_ONSETS = {
    "b": {"b"},
    "c": {"c", "s"},
    "d": {"d"},
    "e": {"", "y"},
    "g": {"g", "j"},
    "p": {"p"},
    "t": {"t"},
    "v": {"v"},
    "z": {"z", "zh"},
}


HIGH_FRONT_VOWEL_TAILS = ("ee", "ii", "ih", "ey", "e", "i", "y")


KNOWN_ASR_CONFUSIONS = {
    "z": {"they", "the", "see", "c"},
    "c": {"see", "sea"},
    "d": {"they"},
    "g": {"gee", "jee"},
    "v": {"they"},
    "x": {"ex", "axe"},
    "y": {"why"},
    "u": {"you"},
    "q": {"cue", "queue"},
    "b": {"bee", "be"},
    "t": {"tea", "tee"},
    "ten": {"then"},
    "then": {"ten"},
    "thin": {"tin"},
    "tin": {"thin"},
    "red": {"read"},
    "read": {"red"},
    "tree": {"three"},
    "three": {"tree"},
    "to": {"two", "too"},
    "two": {"to", "too"},
    "too": {"to", "two"},
    "see": {"sea", "c"},
    "sea": {"see", "c"},
    "bee": {"be", "b"},
    "be": {"bee", "b"},
    "there": {"their", "theyre"},
    "their": {"there", "theyre"},
    "theyre": {"there", "their"},
    "right": {"write"},
    "write": {"right"},
    "hear": {"here"},
    "here": {"hear"},
    "one": {"won"},
    "won": {"one"},
    "four": {"for"},
    "for": {"four"},
    "ate": {"eight"},
    "eight": {"ate"},
    "blue": {"blew"},
    "blew": {"blue"},
    "sun": {"son"},
    "son": {"sun"},
    "i": {"eye"},
    "eye": {"i"},
    "you": {"u"},
    "why": {"y"},
}


KNOWN_CONFUSION_SCORE_OVERRIDES = {
    ("d", "they"): 0.86,
    ("v", "they"): 0.86,
    ("z", "they"): 0.90,
    ("z", "the"): 0.86,
    ("z", "see"): 0.86,
    ("z", "c"): 0.85,
}


@dataclass(frozen=True)
class TranscriptNormalizationResult:
    raw_transcript: str
    corrected_transcript: str
    displayed_transcript: str
    expected_text: str
    raw_wer: float
    corrected_wer: float
    phonetic_similarity_score: float
    normalization_applied: bool
    normalization_reason: str
    correction_strategy_used: str
    accepted_by_phonetic_threshold: bool
    accepted_by_known_confusion: bool
    accepted_by_letter_lattice: bool
    accepted_by_letter_normalization: bool
    accepted_by_exact_match: bool
    threshold_used: float
    confidence_or_threshold_used: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_asr_transcript(
    raw_transcript: str,
    expected_text: str,
    activity_type: str | None = None,
    prompt_type: str | None = None,
    asr_confidence: float | None = None,
    cmudict_loader: CMUDictLoader | None = None,
    config: dict[str, Any] | None = None,
) -> TranscriptNormalizationResult:
    del activity_type, prompt_type

    raw = str(raw_transcript or "").strip()
    expected = str(expected_text or "").strip()
    normalized_raw = normalize_for_wer(raw)
    normalized_expected = normalize_for_wer(expected)
    active_config = config or {}
    default_threshold = _config_float(active_config, ["phonetic_accept_threshold", "high_similarity_threshold"], DEFAULT_PHONETIC_ACCEPT_THRESHOLD)
    strict_word_threshold = _config_float(active_config, ["phonetic_strict_word_threshold", "strict_word_threshold"], DEFAULT_STRICT_WORD_THRESHOLD)
    single_letter_threshold = _config_float(active_config, ["phonetic_single_letter_threshold", "single_letter_threshold"], DEFAULT_SINGLE_LETTER_THRESHOLD)
    known_confusion_threshold = _config_float(active_config, ["phonetic_known_confusion_threshold", "known_confusion_threshold"], DEFAULT_KNOWN_CONFUSION_THRESHOLD)
    lattice_threshold = _config_float(active_config, ["phonetic_lattice_threshold", "lattice_threshold"], DEFAULT_PHONETIC_LATTICE_THRESHOLD)
    low_confidence_threshold = float(active_config.get("low_confidence_threshold", DEFAULT_LOW_CONFIDENCE_THRESHOLD))
    low_confidence_similarity_threshold = float(active_config.get("low_confidence_similarity_threshold", DEFAULT_LOW_CONFIDENCE_SIMILARITY_THRESHOLD))

    score = phonetic_similarity_score(normalized_expected, normalized_raw, cmudict_loader)
    corrected = raw
    displayed = raw
    applied = False
    accepted_for_display = False
    accepted_by_threshold = False
    accepted_by_known_confusion = False
    accepted_by_letter_lattice = False
    accepted_by_letter_normalization = False
    accepted_by_exact_match = False
    threshold_used = default_threshold
    reason = "No normalization applied"
    strategy = "none"

    if not expected:
        reason = "Expected text was empty"
    elif not raw:
        reason = "Raw transcript was empty"
    else:
        is_single_letter = _is_single_letter(normalized_expected)
        is_single_word = _is_single_word(normalized_expected)
        is_word_level = is_single_letter or is_single_word
        threshold_used = (
            single_letter_threshold
            if is_single_letter
            else _word_threshold(normalized_expected, normalized_raw, default_threshold, strict_word_threshold)
        )
        letter_match = _single_letter_correction(normalized_expected, normalized_raw)
        lattice_score = _letter_lattice_score(normalized_expected, normalized_raw) if is_single_letter else 0.0
        if not is_word_level:
            reason = "Expected text is sentence-level or multi-word; word-level transcript correction was skipped"
            threshold_used = 0.0
        elif letter_match:
            corrected = expected
            displayed = expected
            score = 1.0
            applied = raw != expected
            accepted_for_display = True
            accepted_by_letter_normalization = True
            reason = "ASR transcript is a valid spoken form of the expected letter"
            strategy = "letter_pronunciation_normalization"
        elif is_single_letter and lattice_score >= lattice_threshold:
            corrected = expected
            displayed = expected
            score = max(score, lattice_score)
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            accepted_by_letter_lattice = True
            threshold_used = lattice_threshold
            reason = f"Raw ASR output matched generated phonetic spelling variant for expected letter {expected}"
            strategy = "expected_centric_phonetic_lattice_matching"
        elif normalize_transcript(raw) == normalize_transcript(expected):
            corrected = expected
            displayed = expected
            score = 1.0
            applied = raw != expected
            accepted_for_display = True
            accepted_by_exact_match = True
            reason = "ASR transcript already matches expected text after transcript normalization"
            strategy = "orthographic_expected_prompt_alignment"
        elif _known_confusion_match(normalized_expected, normalized_raw):
            threshold_used = known_confusion_threshold if not is_single_letter else single_letter_threshold
            score = max(score, _known_confusion_score(normalized_expected, normalized_raw))
            if score >= threshold_used:
                corrected = expected
                displayed = expected
                applied = True
                accepted_for_display = True
                accepted_by_threshold = True
                accepted_by_known_confusion = True
                reason = (
                    "ASR output passed expected-prompt phonetic similarity threshold for single-letter prompt"
                    if is_single_letter
                    else "ASR transcript is a known homophone or near-homophone of expected text"
                )
                strategy = "letter_phonetic_threshold_alignment" if is_single_letter else "known_confusion_expected_prompt_alignment"
            else:
                reason = "Known ASR confusion did not meet the configured threshold"
        elif is_single_letter:
            reason = "Raw transcript is not a valid spoken form of the expected letter"
        elif not _is_single_word(normalized_raw):
            reason = "Raw transcript is not a single word, so word-level correction was skipped"
        elif _passes_phonetic_threshold(score, threshold_used, asr_confidence, low_confidence_threshold, low_confidence_similarity_threshold):
            corrected = expected
            displayed = expected
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            reason = "Raw ASR output passed word-level phonetic similarity threshold against expected CSV answer"
            strategy = "word_phonetic_threshold_alignment"
        else:
            reason = "Phonetic similarity is too low for safe expected-prompt correction"

    if not accepted_for_display:
        displayed = corrected

    return TranscriptNormalizationResult(
        raw_transcript=raw,
        corrected_transcript=corrected,
        displayed_transcript=displayed,
        expected_text=expected,
        raw_wer=compute_wer(expected, raw),
        corrected_wer=compute_wer(expected, corrected),
        phonetic_similarity_score=round(score, 6),
        normalization_applied=applied,
        normalization_reason=reason,
        correction_strategy_used=strategy,
        accepted_by_phonetic_threshold=accepted_by_threshold,
        accepted_by_known_confusion=accepted_by_known_confusion,
        accepted_by_letter_lattice=accepted_by_letter_lattice,
        accepted_by_letter_normalization=accepted_by_letter_normalization,
        accepted_by_exact_match=accepted_by_exact_match,
        threshold_used=round(threshold_used, 6),
        confidence_or_threshold_used=round(
            low_confidence_similarity_threshold
            if asr_confidence is not None and asr_confidence < low_confidence_threshold
            else threshold_used,
            6,
        ),
    )


def phonetic_similarity_score(
    expected_text: str,
    actual_text: str,
    cmudict_loader: CMUDictLoader | None = None,
) -> float:
    expected_tokens = _tokens(expected_text)
    actual_tokens = _tokens(actual_text)

    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0

    expected_phones = _text_to_phonemes(expected_tokens, cmudict_loader)
    actual_phones = _text_to_phonemes(actual_tokens, cmudict_loader)
    if expected_phones and actual_phones:
        return phoneme_similarity(expected_phones, actual_phones)

    return 1.0 if expected_tokens == actual_tokens else 0.0


def _passes_phonetic_threshold(
    score: float,
    threshold: float,
    asr_confidence: float | None,
    low_confidence_threshold: float,
    low_confidence_similarity_threshold: float,
) -> bool:
    if score >= 1.0:
        return True
    if asr_confidence is not None and asr_confidence < low_confidence_threshold:
        return score >= low_confidence_similarity_threshold
    return score >= threshold


def _config_float(config: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        if key in config:
            return float(config[key])
    return default


def _letter_lattice_score(normalized_expected: str, normalized_raw: str) -> float:
    if not _is_single_letter(normalized_expected):
        return 0.0

    expected_letter = normalized_expected.lower()
    raw = _compact_letter_token(normalized_raw)
    if not raw:
        return 0.0

    core_variants = {_compact_letter_token(variant) for variant in LETTER_PRONUNCIATIONS.get(expected_letter, set())}
    lattice_variants = {_compact_letter_token(variant) for variant in LETTER_LATTICE_VARIANTS.get(expected_letter, set())}

    if raw in core_variants:
        return 0.98
    if raw in lattice_variants:
        return 0.93

    expected_codes = {
        normalize_letter_name_variant(variant)
        for variant in core_variants | lattice_variants
        if normalize_letter_name_variant(variant)
    }
    raw_code = normalize_letter_name_variant(raw)
    if raw_code and raw_code in expected_codes:
        return 0.90

    skeletons = {_consonant_skeleton(variant) for variant in core_variants | lattice_variants}
    raw_skeleton = _consonant_skeleton(raw)
    if raw_skeleton and raw_skeleton in skeletons and _has_similar_vowel_tail(expected_letter, raw):
        return 0.88

    best_edit_similarity = max(
        (SequenceMatcher(a=raw, b=variant).ratio() for variant in core_variants | lattice_variants),
        default=0.0,
    )
    return round(min(0.80, best_edit_similarity), 6)


def normalize_letter_name_variant(token: str) -> str:
    compact = _compact_letter_token(token)
    if not compact:
        return ""

    compact = _limit_repeated_chars(compact)
    for tail in sorted(HIGH_FRONT_VOWEL_TAILS, key=len, reverse=True):
        if compact.endswith(tail):
            onset = compact[: -len(tail)]
            return f"{onset}#IY"

    return compact


def _compact_letter_token(text: str) -> str:
    return normalize_transcript(text).replace("-", " ").replace(" ", "")


def _limit_repeated_chars(token: str, max_repeats: int = 2) -> str:
    compacted: list[str] = []
    previous = ""
    count = 0
    for char in token:
        if char == previous:
            count += 1
        else:
            previous = char
            count = 1
        if count <= max_repeats:
            compacted.append(char)
    return "".join(compacted)


def _consonant_skeleton(token: str) -> str:
    compact = _compact_letter_token(token)
    return "".join(char for char in compact if char not in {"a", "e", "i", "o", "u", "y"})


def _has_similar_vowel_tail(expected_letter: str, raw: str) -> bool:
    allowed_onsets = LETTER_HIGH_FRONT_VOWEL_ONSETS.get(expected_letter, set())
    raw_code = normalize_letter_name_variant(raw)
    if not raw_code.endswith("#IY"):
        return False
    return raw_code.removesuffix("#IY") in allowed_onsets


def _single_letter_correction(normalized_expected: str, normalized_raw: str) -> bool:
    if not _is_single_letter(normalized_expected):
        return False

    expected_letter = normalized_expected.lower()
    raw = normalized_raw.replace("-", " ")
    compact_raw = raw.replace(" ", "")
    spoken_forms = LETTER_PRONUNCIATIONS.get(expected_letter, set())

    return raw in spoken_forms or compact_raw in spoken_forms


def _is_single_letter(normalized_expected: str) -> bool:
    return len(normalized_expected) == 1 and normalized_expected.isalpha()


def _is_single_word(normalized_text: str) -> bool:
    return bool(normalized_text) and len(normalized_text.split()) == 1 and any(char.isalpha() for char in normalized_text)


def _word_threshold(normalized_expected: str, normalized_raw: str, default_threshold: float, strict_word_threshold: float) -> float:
    if not _is_single_word(normalized_raw):
        return default_threshold
    if normalized_expected[:1] and normalized_raw[:1] and normalized_expected[:1] != normalized_raw[:1]:
        return strict_word_threshold
    return default_threshold


def _known_confusion_match(normalized_expected: str, normalized_raw: str) -> bool:
    if not normalized_expected or not normalized_raw:
        return False

    expected = normalized_expected.replace("'", "")
    raw = normalized_raw.replace("'", "")
    if expected == raw:
        return True

    return raw in KNOWN_ASR_CONFUSIONS.get(expected, set())


def _known_confusion_score(normalized_expected: str, normalized_raw: str) -> float:
    expected = normalized_expected.replace("'", "")
    raw = normalized_raw.replace("'", "")
    return KNOWN_CONFUSION_SCORE_OVERRIDES.get((expected, raw), 0.95)


def _text_to_phonemes(tokens: list[str], cmudict_loader: CMUDictLoader | None) -> list[str]:
    active_loader = cmudict_loader or CMUDictLoader().load()
    phonemes: list[str] = []
    for token in tokens:
        pronunciation = active_loader.get_primary_pronunciation(token)
        if not pronunciation:
            return []
        phonemes.extend(pronunciation)
    return phonemes


def _tokens(text: str) -> list[str]:
    normalized = normalize_for_wer(text)
    return [] if normalized == "" else normalized.split(" ")
