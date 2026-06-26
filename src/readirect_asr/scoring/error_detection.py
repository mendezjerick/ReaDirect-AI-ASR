from __future__ import annotations

from typing import Any

from readirect_asr.scoring.answer_matching import match_answer, parse_accepted_answers
from readirect_asr.scoring.feedback_hints import generate_feedback_hint
from readirect_asr.scoring.phoneme_comparison import compare_phonemes
from readirect_asr.scoring.skill_signals import infer_skill_signal


def detect_error_type(
    expected_text: str,
    actual_text: str,
    accepted_answers: Any = None,
    expected_phonemes: list[str] | None = None,
    actual_phonemes: list[str] | None = None,
    similarity_label: str | None = None,
) -> dict[str, object]:
    match = match_answer(expected_text, actual_text, parse_accepted_answers(accepted_answers))
    expected_norm = str(match["normalized_expected"])
    actual_norm = str(match["normalized_actual"])
    expected_tokens = expected_norm.split()
    actual_tokens = actual_norm.split()
    label = similarity_label or str(match["similarity_label"])
    phoneme_report = compare_phonemes(expected_phonemes or [], actual_phonemes or [])

    if not actual_norm:
        return _result("blank", None, expected_text, expected_phonemes, "No transcript or answer was available.")
    if match["is_exact"]:
        return _result("correct", None, expected_text, expected_phonemes, "Normalized expected and actual text matched.")
    if match["is_accepted"]:
        return _result("accepted_variant", None, expected_text, expected_phonemes, "Actual answer matched an accepted variant.")

    if len(expected_tokens) == 1 and len(actual_tokens) == 1:
        single = _detect_single_word_error(expected_text, phoneme_report, expected_phonemes or [], actual_phonemes or [], label)
        if single:
            return single

    if len(expected_tokens) > 1:
        sentence = _detect_sentence_error(expected_tokens, actual_tokens, label, expected_text, expected_phonemes or [])
        if sentence:
            return sentence

    if label == "very_close":
        return _result("very_close_text_error", None, expected_text, expected_phonemes, "Text was very close but not an accepted answer.")
    if label == "far":
        return _result("far_answer", None, expected_text, expected_phonemes, "Actual answer had low similarity to expected text.")
    return _result("incorrect_general", None, expected_text, expected_phonemes, "No specific heuristic error type was identified.")


def _detect_single_word_error(
    expected_text: str,
    phoneme_report: dict[str, object],
    expected_phonemes: list[str],
    actual_phonemes: list[str],
    similarity_label: str,
) -> dict[str, object] | None:
    initial = phoneme_report["initial_phoneme_match"]
    final = phoneme_report["final_phoneme_match"]
    vowel = phoneme_report["vowel_phoneme_match"]
    if expected_phonemes and actual_phonemes:
        if initial is False and vowel is not False and final is not False:
            return _result("initial_sound_error", "initial", expected_text, expected_phonemes, "Final and vowel phonemes mostly matched, but initial phoneme differed.")
        if final is False and initial is not False and vowel is not False:
            return _result("final_sound_error", "final", expected_text, expected_phonemes, "Initial and vowel phonemes matched, but final phoneme differed.")
        if vowel is False and initial is not False and final is not False:
            return _result("vowel_error", "medial", expected_text, expected_phonemes, "Initial and final phonemes mostly matched, but vowel phoneme differed.")
        if len(actual_phonemes) < len(expected_phonemes):
            return _result("omission", None, expected_text, expected_phonemes, "Actual phoneme sequence was shorter than expected.")
        if len(actual_phonemes) > len(expected_phonemes):
            return _result("insertion", None, expected_text, expected_phonemes, "Actual phoneme sequence was longer than expected.")
        if phoneme_report["phoneme_similarity"] and float(phoneme_report["phoneme_similarity"]) >= 0.5:
            return _result("consonant_error", None, expected_text, expected_phonemes, "Phoneme sequence was close but contained a consonant mismatch.")
    if similarity_label in {"very_close", "close"}:
        return _result("substitution", None, expected_text, expected_phonemes, "Single-word answer was similar but substituted one or more sounds.")
    return None


def _detect_sentence_error(
    expected_tokens: list[str],
    actual_tokens: list[str],
    similarity_label: str,
    expected_text: str,
    expected_phonemes: list[str],
) -> dict[str, object] | None:
    if not actual_tokens:
        return _result("blank", None, expected_text, expected_phonemes, "No sentence transcript was available.")
    if len(actual_tokens) <= max(1, len(expected_tokens) // 2):
        return _result("partial_sentence", None, expected_text, expected_phonemes, "Actual sentence was much shorter than expected.")
    if sorted(expected_tokens) == sorted(actual_tokens) and expected_tokens != actual_tokens:
        return _result("word_order_error", None, expected_text, expected_phonemes, "Expected words were present but in a different order.")
    missing = [token for token in expected_tokens if token not in actual_tokens]
    if missing:
        return _result("skipped_word", None, expected_text, expected_phonemes, "One or more expected words were missing.")
    if len(actual_tokens) > len(expected_tokens):
        return _result("insertion", None, expected_text, expected_phonemes, "Actual sentence included extra words.")
    if similarity_label == "far":
        return _result("far_answer", None, expected_text, expected_phonemes, "Actual sentence had low similarity to expected sentence.")
    return None


def _result(
    error_type: str,
    error_position: str | None,
    expected_text: str,
    expected_phonemes: list[str] | None,
    explanation: str,
) -> dict[str, object]:
    skill = infer_skill_signal(error_type, expected_text, expected_phonemes)
    feedback = generate_feedback_hint(error_type, str(skill["skill_signal"]))
    return {
        "error_type": error_type,
        "error_position": error_position,
        "feedback_hint": feedback["feedback_hint"],
        "skill_signal": skill["skill_signal"],
        "confidence": "heuristic",
        "explanation": explanation,
    }
