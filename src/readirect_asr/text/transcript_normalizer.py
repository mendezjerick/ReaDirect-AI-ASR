from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.phoneme_comparison import phoneme_similarity
from readirect_asr.text.normalization import normalize_for_wer, normalize_transcript
from readirect_asr.text.reinforcement_corrections import normalize_prompt_type, reinforcement_table_from_config


DEFAULT_PHONETIC_ACCEPT_THRESHOLD = 0.88
DEFAULT_STRICT_WORD_THRESHOLD = 0.90
DEFAULT_SINGLE_LETTER_THRESHOLD = 0.85
DEFAULT_KNOWN_CONFUSION_THRESHOLD = 0.82
DEFAULT_PHONETIC_LATTICE_THRESHOLD = 0.85
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_LOW_CONFIDENCE_SIMILARITY_THRESHOLD = 0.95


LETTER_PRONUNCIATIONS = {
    "a": {"a", "ay", "aye", "ai", "ey"},
    "b": {"b", "bee", "be", "bi", "by", "bii"},
    "c": {"c", "see", "sea", "si", "cy", "sii", "s"},
    "d": {"d", "dee", "de", "di", "dy", "dii"},
    "e": {"e", "ee", "i", "ii", "yi"},
    "f": {"f", "ef", "eff"},
    "g": {"g", "gee", "jee", "gi", "gy", "gii", "j"},
    "h": {"h", "aitch", "haitch", "age", "each"},
    "i": {"i", "eye", "ai", "ay"},
    "j": {"j", "jay", "jai", "jey"},
    "k": {"k", "kay", "kei", "key"},
    "l": {"l", "el", "ell", "elle"},
    "m": {"m", "em", "emm"},
    "n": {"n", "en", "enn"},
    "o": {"o", "oh", "owe"},
    "p": {"p", "pee", "pi", "py", "pii"},
    "q": {"q", "cue", "queue", "kyoo", "kyu"},
    "r": {"r", "are", "ar"},
    "s": {"s", "ess", "es"},
    "t": {"t", "tee", "tea", "ti", "ty", "tii"},
    "u": {"u", "you", "yu", "yoo"},
    "v": {"v", "vee", "vi", "vy", "vii"},
    "w": {"w", "double you", "double u", "doubleu", "double-u", "dubya"},
    "x": {"x", "ex", "axe", "eks"},
    "y": {"y", "why", "wai", "wy"},
    "z": {"z", "zee", "zed", "zi", "zii", "zih", "zy", "zey", "ze", "zhee", "they", "the", "see"},
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
    "they're": {"there", "their"},
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


LETTER_PHONEMES = {
    "A": ["EY"],
    "B": ["B", "IY"],
    "C": ["S", "IY"],
    "D": ["D", "IY"],
    "E": ["IY"],
    "F": ["EH", "F"],
    "G": ["JH", "IY"],
    "H": ["EY", "CH"],
    "I": ["AY"],
    "J": ["JH", "EY"],
    "K": ["K", "EY"],
    "L": ["EH", "L"],
    "M": ["EH", "M"],
    "N": ["EH", "N"],
    "O": ["OW"],
    "P": ["P", "IY"],
    "Q": ["K", "Y", "UW"],
    "R": ["AA", "R"],
    "S": ["EH", "S"],
    "T": ["T", "IY"],
    "U": ["Y", "UW"],
    "V": ["V", "IY"],
    "W": ["D", "AH", "B", "AH", "L", "Y", "UW"],
    "X": ["EH", "K", "S"],
    "Y": ["W", "AY"],
    "Z": ["Z", "IY"],
}


WORD_PHONEME_OVERRIDES = {
    "ten": ["T", "EH", "N"],
    "then": ["DH", "EH", "N"],
    "thin": ["TH", "IH", "N"],
    "tin": ["T", "IH", "N"],
    "tree": ["T", "R", "IY"],
    "three": ["TH", "R", "IY"],
    "red": ["R", "EH", "D"],
    "read": ["R", "EH", "D"],
    "to": ["T", "UW"],
    "two": ["T", "UW"],
    "too": ["T", "UW"],
    "see": ["S", "IY"],
    "sea": ["S", "IY"],
    "bee": ["B", "IY"],
    "be": ["B", "IY"],
    "right": ["R", "AY", "T"],
    "write": ["R", "AY", "T"],
    "hear": ["HH", "IY", "R"],
    "here": ["HH", "IY", "R"],
    "one": ["W", "AH", "N"],
    "won": ["W", "AH", "N"],
    "four": ["F", "AO", "R"],
    "for": ["F", "AO", "R"],
    "ate": ["EY", "T"],
    "eight": ["EY", "T"],
    "sun": ["S", "AH", "N"],
    "son": ["S", "AH", "N"],
    "there": ["DH", "EH", "R"],
    "their": ["DH", "EH", "R"],
    "theyre": ["DH", "EH", "R"],
}


CRITICAL_PHONEME_RULES = {
    "Q": {"pair": "U", "critical": "K", "position": "initial", "reason": "Q requires initial K before Y UW"},
    "U": {"pair": "Q", "critical": "Y", "position": "initial", "reason": "U starts with Y UW without initial K"},
    "C": {"pair": "Z", "critical": "S", "position": "initial", "reason": "C requires S rather than Z"},
    "Z": {"pair": "C", "critical": "Z", "position": "initial", "reason": "Z requires Z rather than S"},
    "B": {"pair": "V", "critical": "B", "position": "initial", "reason": "B requires B rather than V"},
    "V": {"pair": "B", "critical": "V", "position": "initial", "reason": "V requires V rather than B"},
    "D": {"pair": "T", "critical": "D", "position": "initial", "reason": "D requires D rather than T"},
    "T": {"pair": "D", "critical": "T", "position": "initial", "reason": "T requires T rather than D"},
    "M": {"pair": "N", "critical": "M", "position": "final", "reason": "M requires final M rather than N"},
    "N": {"pair": "M", "critical": "N", "position": "final", "reason": "N requires final N rather than M"},
    "F": {"pair": "S", "critical": "F", "position": "final", "reason": "F requires final F rather than S"},
    "S": {"pair": "F", "critical": "S", "position": "final", "reason": "S requires final S rather than F"},
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
    prompt_type: str
    asr_route: str
    model_family: str
    model_used: str
    wav2vec2_transcript: str
    whisper_transcript: None
    whisper_removed: bool
    raw_wer: float
    corrected_wer: float
    raw_cer: float
    corrected_cer: float
    expected_phonemes: list[str]
    expected_phoneme_source: str
    expected_phoneme_variants: list[list[str]]
    observed_phonemes: list[str]
    phonetic_similarity_score: float
    composite_score: float
    accepted: bool
    normalization_applied: bool
    normalization_reason: str
    correction_strategy_used: str
    accepted_by_letter_alias: bool
    accepted_by_phonetic_threshold: bool
    accepted_by_known_confusion: bool
    accepted_by_letter_lattice: bool
    accepted_by_letter_normalization: bool
    accepted_by_exact_match: bool
    accepted_by_vowel_tail: bool
    accepted_by_phoneme_evidence: bool
    accepted_by_reinforcement_match: bool
    reinforcement_source_file: str
    reinforcement_expected_label: str
    reinforcement_matched_transcript: str
    reinforcement_match_normalized: dict[str, Any]
    reinforcement_match_original: dict[str, Any]
    critical_phoneme: str | None
    critical_phoneme_detected: bool | None
    critical_phoneme_expected_position: str | None
    critical_phoneme_reason: str | None
    critical_pair_detected: bool
    confidence_level: str
    threshold_used: float
    confidence_or_threshold_used: float
    debug_metadata: dict[str, Any]
    gop_enabled: bool = True
    gop_available: bool = False
    gop_score: float | None = None
    gop_confidence: float | None = None
    gop_decision: str = "not_available"
    gop_threshold: float | None = None
    gop_prompt_type: str = "unknown"
    gop_expected_phonemes: list[str] | None = None
    gop_observed_phonemes: list[str] | None = None
    gop_phoneme_scores: list[dict[str, Any]] | None = None
    gop_word_scores: list[dict[str, Any]] | None = None
    mispronounced_phonemes: list[str] | None = None
    weak_words: list[str] | None = None
    gop_correction_applied: bool = False
    gop_error: str | None = None
    dynamic_correction_enabled: bool = True
    dynamic_correction_applied: bool = False
    dynamic_correction_strategy: str = "dynamic_expected_word_correction"
    dynamic_correction_sub_strategy: str = ""
    dynamic_correction_confidence: float | None = None
    dynamic_correction_threshold: float | None = None
    dynamic_spelling_similarity: float | None = None
    dynamic_phoneme_similarity: float | None = None
    dynamic_gop_score: float | None = None
    dynamic_homophone_match: bool = False
    dynamic_context_score: float | None = None
    dynamic_correction_reason: str = ""
    word_alignment: list[dict[str, Any]] | None = None

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
    observed_phonemes: list[str] | None = None,
    wav2vec2_transcript: str | None = None,
    model_used: str = "",
    asr_route: str = "wav2vec2_only",
) -> TranscriptNormalizationResult:
    raw = str(raw_transcript or "").strip()
    expected = str(expected_text or "").strip()
    normalized_raw = normalize_for_wer(raw)
    normalized_expected = normalize_for_wer(expected)
    detected_prompt_type = detect_prompt_type(expected, activity_type=activity_type, prompt_type=prompt_type)
    normalized_observed_phonemes = _normalize_phonemes(observed_phonemes or [])
    expected_phonemes, expected_phoneme_source, expected_phoneme_variants = generate_expected_phonemes(expected, cmudict_loader)
    active_config = config or {}
    default_threshold = _config_float(active_config, ["phonetic_accept_threshold", "high_similarity_threshold"], DEFAULT_PHONETIC_ACCEPT_THRESHOLD)
    strict_word_threshold = _config_float(active_config, ["phonetic_strict_word_threshold", "strict_word_threshold"], DEFAULT_STRICT_WORD_THRESHOLD)
    single_letter_threshold = _config_float(active_config, ["phonetic_single_letter_threshold", "single_letter_threshold"], DEFAULT_SINGLE_LETTER_THRESHOLD)
    known_confusion_threshold = _config_float(active_config, ["phonetic_known_confusion_threshold", "known_confusion_threshold"], DEFAULT_KNOWN_CONFUSION_THRESHOLD)
    lattice_threshold = _config_float(active_config, ["phonetic_lattice_threshold", "lattice_threshold"], DEFAULT_PHONETIC_LATTICE_THRESHOLD)
    low_confidence_threshold = float(active_config.get("low_confidence_threshold", DEFAULT_LOW_CONFIDENCE_THRESHOLD))
    low_confidence_similarity_threshold = float(active_config.get("low_confidence_similarity_threshold", DEFAULT_LOW_CONFIDENCE_SIMILARITY_THRESHOLD))
    critical_required = _as_bool(active_config.get("critical_phoneme_required", True))

    score = phonetic_similarity_score(normalized_expected, normalized_raw, cmudict_loader)
    phoneme_score = phoneme_similarity(expected_phonemes, normalized_observed_phonemes) if expected_phonemes and normalized_observed_phonemes else 0.0
    corrected = raw
    displayed = raw
    applied = False
    accepted_for_display = False
    accepted_by_threshold = False
    accepted_by_known_confusion = False
    accepted_by_letter_lattice = False
    accepted_by_letter_normalization = False
    accepted_by_exact_match = False
    accepted_by_letter_alias = False
    accepted_by_vowel_tail = False
    accepted_by_phoneme_evidence = False
    accepted_by_reinforcement_match = False
    reinforcement_source_file = ""
    reinforcement_expected_label = ""
    reinforcement_matched_transcript = ""
    reinforcement_match_normalized: dict[str, Any] = {}
    reinforcement_match_original: dict[str, Any] = {}
    threshold_used = default_threshold
    reason = "No normalization applied"
    strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
    critical = _critical_phoneme_check(normalized_expected, normalized_observed_phonemes)
    confidence_level = "normal"
    composite_score = max(score, phoneme_score)

    if not expected:
        reason = "Expected text was empty"
        strategy = "none"
    else:
        is_single_letter = detected_prompt_type == "letter"
        is_single_word = detected_prompt_type in {"word", "rhyme", "rhyming_word"}
        is_word_level = is_single_letter or is_single_word
        threshold_used = (
            single_letter_threshold
            if is_single_letter
            else _word_threshold(normalized_expected, normalized_raw, default_threshold, strict_word_threshold)
        )
        letter_match = _single_letter_correction(normalized_expected, normalized_raw)
        lattice_score = _letter_lattice_score(normalized_expected, normalized_raw) if is_single_letter else 0.0
        exact_match = normalize_transcript(raw) == normalize_transcript(expected)
        reinforcement_table = reinforcement_table_from_config(active_config)
        reinforcement_match = reinforcement_table.match(expected, raw, detected_prompt_type)
        composite_score = _composite_score(
            text_score=score,
            phoneme_score=phoneme_score,
            lattice_score=lattice_score,
            exact_match=exact_match,
            letter_alias_match=letter_match,
            known_confusion_match=_known_confusion_match(normalized_expected, normalized_raw),
            critical=critical,
        )
        if exact_match:
            corrected = expected
            displayed = expected
            score = 1.0
            composite_score = 1.0
            applied = raw != expected
            accepted_for_display = True
            accepted_by_exact_match = True
            reason = "ASR transcript already matches expected text after transcript normalization"
            strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
        elif _critical_blocks_acceptance(critical, critical_required, normalized_observed_phonemes):
            confidence_level = "low"
            reason = critical["critical_phoneme_reason"] or "Critical phoneme evidence contradicted expected answer"
        elif reinforcement_match and is_single_letter:
            match_metadata = reinforcement_match.to_metadata()
            corrected = expected
            displayed = expected
            score = 1.0
            composite_score = 1.0
            applied = True
            accepted_for_display = True
            accepted_by_reinforcement_match = True
            reinforcement_source_file = str(match_metadata["reinforcement_source_file"])
            reinforcement_expected_label = str(match_metadata["reinforcement_expected_label"])
            reinforcement_matched_transcript = str(match_metadata["reinforcement_matched_transcript"])
            reinforcement_match_normalized = dict(match_metadata["reinforcement_match_normalized"])
            reinforcement_match_original = dict(match_metadata["reinforcement_match_original"])
            reason = (
                "Raw transcript matched curated reinforcement correction row: "
                f"expected {reinforcement_expected_label}, transcript-error {reinforcement_matched_transcript}"
            )
            strategy = "reinforcement_error_transcript_match"
        elif not is_word_level:
            reason = "Expected text is sentence-level or multi-word; word-level transcript correction was skipped"
            threshold_used = 0.0
            strategy = "wav2vec2_sentence_wer_cer_scoring" if detected_prompt_type in {"sentence", "paragraph", "passage", "final_sentence", "reading_passage"} else "none"
        elif letter_match:
            corrected = expected
            displayed = expected
            score = 1.0
            composite_score = 1.0
            applied = raw != expected
            accepted_for_display = True
            accepted_by_letter_normalization = True
            accepted_by_letter_alias = True
            accepted_by_letter_lattice = _letter_lattice_variant_match(normalized_expected, normalized_raw)
            accepted_by_known_confusion = _known_confusion_match(normalized_expected, normalized_raw)
            accepted_by_threshold = accepted_by_letter_lattice or accepted_by_known_confusion
            reason = "ASR transcript is a valid spoken form of the expected letter"
            strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
        elif is_single_letter and lattice_score >= lattice_threshold:
            corrected = expected
            displayed = expected
            score = max(score, lattice_score)
            composite_score = max(composite_score, lattice_score)
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            accepted_by_letter_lattice = True
            threshold_used = lattice_threshold
            reason = f"Raw ASR output matched generated phonetic spelling variant for expected letter {expected}"
            strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
        elif _known_confusion_match(normalized_expected, normalized_raw):
            threshold_used = known_confusion_threshold if not is_single_letter else single_letter_threshold
            score = max(score, _known_confusion_score(normalized_expected, normalized_raw))
            composite_score = max(composite_score, score)
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
                strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
            else:
                reason = "Known ASR confusion did not meet the configured threshold"
        elif reinforcement_match:
            match_metadata = reinforcement_match.to_metadata()
            corrected = expected
            displayed = expected
            score = 1.0
            composite_score = 1.0
            applied = True
            accepted_for_display = True
            accepted_by_reinforcement_match = True
            reinforcement_source_file = str(match_metadata["reinforcement_source_file"])
            reinforcement_expected_label = str(match_metadata["reinforcement_expected_label"])
            reinforcement_matched_transcript = str(match_metadata["reinforcement_matched_transcript"])
            reinforcement_match_normalized = dict(match_metadata["reinforcement_match_normalized"])
            reinforcement_match_original = dict(match_metadata["reinforcement_match_original"])
            reason = (
                "Raw transcript matched curated reinforcement correction row: "
                f"expected {reinforcement_expected_label}, transcript-error {reinforcement_matched_transcript}"
            )
            strategy = "reinforcement_error_transcript_match"
        elif is_single_letter and phoneme_score >= single_letter_threshold:
            corrected = expected
            displayed = expected
            score = max(score, phoneme_score)
            composite_score = max(composite_score, phoneme_score)
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            accepted_by_phoneme_evidence = True
            reason = "Wav2Vec2 text was ambiguous but phoneme evidence matched expected letter"
        elif is_single_letter and not normalized_raw and phoneme_score >= single_letter_threshold:
            corrected = expected
            displayed = expected
            score = max(score, phoneme_score)
            composite_score = max(composite_score, phoneme_score)
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            accepted_by_phoneme_evidence = True
            reason = "Blank Wav2Vec2 text accepted from strong phoneme evidence for expected letter"
        elif is_single_letter:
            reason = "Raw transcript is not a valid spoken form of the expected letter"
        elif not _is_single_word(normalized_raw):
            reason = "Raw transcript is not a single word, so word-level correction was skipped"
        elif _passes_phonetic_threshold(max(score, phoneme_score), threshold_used, asr_confidence, low_confidence_threshold, low_confidence_similarity_threshold):
            corrected = expected
            displayed = expected
            applied = True
            accepted_for_display = True
            accepted_by_threshold = True
            accepted_by_phoneme_evidence = phoneme_score >= threshold_used and phoneme_score >= score
            composite_score = max(composite_score, score, phoneme_score)
            reason = "Raw ASR output passed word-level phonetic similarity threshold against expected CSV answer"
            strategy = "wav2vec2_expected_centric_acoustic_phonetic_scoring"
        else:
            reason = "Phonetic similarity is too low for safe expected-prompt correction"

    if not accepted_for_display:
        displayed = corrected

    return TranscriptNormalizationResult(
        raw_transcript=raw,
        corrected_transcript=corrected,
        displayed_transcript=displayed,
        expected_text=expected,
        prompt_type=detected_prompt_type,
        asr_route=asr_route,
        model_family="wav2vec2",
        model_used=model_used,
        wav2vec2_transcript=wav2vec2_transcript if wav2vec2_transcript is not None else raw,
        whisper_transcript=None,
        whisper_removed=True,
        raw_wer=compute_wer(expected, raw),
        corrected_wer=compute_wer(expected, corrected),
        raw_cer=compute_cer(expected, raw),
        corrected_cer=compute_cer(expected, corrected),
        expected_phonemes=expected_phonemes,
        expected_phoneme_source=expected_phoneme_source,
        expected_phoneme_variants=expected_phoneme_variants,
        observed_phonemes=normalized_observed_phonemes,
        phonetic_similarity_score=round(score, 6),
        composite_score=round(composite_score, 6),
        accepted=accepted_for_display,
        normalization_applied=applied,
        normalization_reason=reason,
        correction_strategy_used=strategy,
        accepted_by_letter_alias=accepted_by_letter_alias,
        accepted_by_phonetic_threshold=accepted_by_threshold,
        accepted_by_known_confusion=accepted_by_known_confusion,
        accepted_by_letter_lattice=accepted_by_letter_lattice,
        accepted_by_letter_normalization=accepted_by_letter_normalization,
        accepted_by_exact_match=accepted_by_exact_match,
        accepted_by_vowel_tail=accepted_by_vowel_tail,
        accepted_by_phoneme_evidence=accepted_by_phoneme_evidence,
        accepted_by_reinforcement_match=accepted_by_reinforcement_match,
        reinforcement_source_file=reinforcement_source_file,
        reinforcement_expected_label=reinforcement_expected_label,
        reinforcement_matched_transcript=reinforcement_matched_transcript,
        reinforcement_match_normalized=reinforcement_match_normalized,
        reinforcement_match_original=reinforcement_match_original,
        critical_phoneme=critical["critical_phoneme"],
        critical_phoneme_detected=critical["critical_phoneme_detected"],
        critical_phoneme_expected_position=critical["critical_phoneme_expected_position"],
        critical_phoneme_reason=critical["critical_phoneme_reason"],
        critical_pair_detected=critical["critical_pair_detected"],
        confidence_level=confidence_level if normalized_observed_phonemes or not critical["critical_pair_detected"] else "lower_without_phoneme_evidence",
        threshold_used=round(threshold_used, 6),
        confidence_or_threshold_used=round(
            low_confidence_similarity_threshold
            if asr_confidence is not None and asr_confidence < low_confidence_threshold
            else threshold_used,
            6,
        ),
        debug_metadata={
            "phoneme_similarity_score": phoneme_score,
            "prompt_type_detection": detected_prompt_type,
            "critical_phoneme_required": critical_required,
            "reinforcement_corrections": reinforcement_table_from_config(active_config).status(),
        },
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


def detect_prompt_type(expected_text: str, activity_type: str | None = None, prompt_type: str | None = None) -> str:
    normalized = normalize_for_wer(expected_text)
    explicit = normalize_prompt_type(prompt_type or activity_type or "")
    if explicit in {"letter", "word", "rhyme", "rhyming_word", "sentence", "paragraph", "passage", "final_sentence", "reading_passage"}:
        return explicit
    if not normalized:
        return "unknown"
    if len(normalized) == 1 and normalized.isalpha():
        return "letter"
    tokens = normalized.split()
    if len(tokens) == 1:
        return "word"
    return "sentence"


def generate_expected_phonemes(expected_text: str, cmudict_loader: CMUDictLoader | None = None) -> tuple[list[str], str, list[list[str]]]:
    expected = str(expected_text or "").strip()
    normalized = normalize_for_wer(expected)
    if len(normalized) == 1 and normalized.isalpha():
        phones = LETTER_PHONEMES.get(normalized.upper(), [])
        return phones, "custom_letter_dictionary", [phones] if phones else []
    tokens = _tokens(expected)
    if not tokens:
        return [], "empty", []
    if len(tokens) > 1:
        return [], "sentence_skipped", []
    override_phones: list[str] = []
    all_overridden = True
    for token in tokens:
        phones = WORD_PHONEME_OVERRIDES.get(token.replace("'", ""))
        if not phones:
            all_overridden = False
            break
        override_phones.extend(phones)
    if all_overridden and override_phones:
        return override_phones, "readirect_overrides", [override_phones]
    variants: list[list[str]] = []
    active_loader = cmudict_loader or _default_cmudict_loader()
    cmu_phones: list[str] = []
    all_cmu_found = True
    for token in tokens:
        pronunciations = active_loader.get_pronunciations(token)
        if pronunciations:
            cmu_phones.extend(pronunciations[0])
            variants.extend(pronunciations[:3])
        else:
            all_cmu_found = False
            break
    if all_cmu_found and cmu_phones:
        return _normalize_phonemes(cmu_phones), "cmudict", [_normalize_phonemes(variant) for variant in variants]
    try:
        from g2p_en import G2p

        g2p = G2p()
        phones = _normalize_phonemes([part for part in g2p(expected) if str(part).strip()])
        if phones:
            return phones, "g2p_en", [phones]
    except Exception:
        pass
    return [], "unavailable", []


def _normalize_phonemes(phonemes: list[str]) -> list[str]:
    normalized: list[str] = []
    for phoneme in phonemes:
        clean = "".join(char for char in str(phoneme or "").upper() if char.isalpha())
        if clean:
            normalized.append(clean)
    return normalized


def _critical_phoneme_check(normalized_expected: str, observed_phonemes: list[str]) -> dict[str, Any]:
    expected = normalized_expected.upper()
    rule = CRITICAL_PHONEME_RULES.get(expected) if len(expected) == 1 else None
    if not rule:
        return {
            "critical_phoneme": None,
            "critical_phoneme_detected": None,
            "critical_phoneme_expected_position": None,
            "critical_phoneme_reason": None,
            "critical_pair_detected": False,
        }
    critical = str(rule["critical"])
    position = str(rule["position"])
    detected = None
    if observed_phonemes:
        if position == "initial":
            detected = observed_phonemes[0] == critical
        elif position == "final":
            detected = observed_phonemes[-1] == critical
        else:
            detected = critical in observed_phonemes
    return {
        "critical_phoneme": critical,
        "critical_phoneme_detected": detected,
        "critical_phoneme_expected_position": position,
        "critical_phoneme_reason": str(rule["reason"]),
        "critical_pair_detected": True,
    }


def _critical_blocks_acceptance(critical: dict[str, Any], required: bool, observed_phonemes: list[str]) -> bool:
    if not required or not critical["critical_pair_detected"] or not observed_phonemes:
        return False
    return critical["critical_phoneme_detected"] is False


def _composite_score(
    text_score: float,
    phoneme_score: float,
    lattice_score: float,
    exact_match: bool,
    letter_alias_match: bool,
    known_confusion_match: bool,
    critical: dict[str, Any],
) -> float:
    score = max(text_score, phoneme_score, lattice_score)
    if exact_match:
        score = max(score, 1.0)
    if letter_alias_match:
        score = max(score, 0.98)
    if lattice_score >= DEFAULT_PHONETIC_LATTICE_THRESHOLD:
        score = max(score, 0.93)
    if known_confusion_match:
        score = max(score, 0.90)
    if critical["critical_phoneme_detected"] is True:
        score = min(1.0, max(score, 0.90) + 0.05)
    elif critical["critical_phoneme_detected"] is False:
        score = min(score, 0.60)
    return round(score, 6)


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _letter_lattice_variant_match(normalized_expected: str, normalized_raw: str) -> bool:
    if not _is_single_letter(normalized_expected):
        return False
    expected_letter = normalized_expected.lower()
    raw = _compact_letter_token(normalized_raw)
    variants = {_compact_letter_token(variant) for variant in LETTER_LATTICE_VARIANTS.get(expected_letter, set())}
    return raw in variants


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
    active_loader = cmudict_loader or _default_cmudict_loader()
    phonemes: list[str] = []
    for token in tokens:
        override = WORD_PHONEME_OVERRIDES.get(token.replace("'", ""))
        if override:
            phonemes.extend(override)
            continue
        pronunciation = active_loader.get_primary_pronunciation(token)
        if not pronunciation:
            return []
        phonemes.extend(pronunciation)
    return phonemes


@lru_cache(maxsize=1)
def _default_cmudict_loader() -> CMUDictLoader:
    return CMUDictLoader().load()


def _tokens(text: str) -> list[str]:
    normalized = normalize_for_wer(text)
    return [] if normalized == "" else normalized.split(" ")
