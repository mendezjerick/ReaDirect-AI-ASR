import math

from api.service import AIAnalysisService
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.pronunciation.gop import apply_gop_to_transcript_meta, canonical_expected_phonemes, compute_gop, ctc_forced_align
from readirect_asr.text.transcript_normalizer import normalize_asr_transcript


VOCAB = {
    0: "<pad>",
    1: "K",
    2: "AE",
    3: "T",
    4: "AH",
    5: "P",
    6: "B",
    7: "L",
    8: "OW",
    9: "G",
    10: "UW",
    11: "HH",
    12: "IY",
    13: "AO",
}


def _log_row(best_id: int, weak_expected_id: int | None = None, competitor_id: int | None = None) -> list[float]:
    probs = [0.01] * len(VOCAB)
    probs[0] = 0.03
    probs[best_id] = 0.90
    if weak_expected_id is not None and competitor_id is not None:
        probs[best_id] = 0.03
        probs[weak_expected_id] = 0.22
        probs[competitor_id] = 0.68
    total = sum(probs)
    return [math.log(value / total) for value in probs]


def _evidence(phone_ids: list[int], *, weak_index: int | None = None, competitor_id: int | None = None) -> dict:
    rows = [_log_row(0)]
    decoded = []
    for index, phone_id in enumerate(phone_ids):
        if weak_index == index and competitor_id is not None:
            rows.extend([_log_row(phone_id, phone_id, competitor_id), _log_row(phone_id, phone_id, competitor_id)])
            decoded.append(VOCAB[competitor_id])
        else:
            rows.extend([_log_row(phone_id), _log_row(phone_id)])
            decoded.append(VOCAB[phone_id])
        rows.append(_log_row(0))
    return {
        "available": True,
        "model_version": "existing_wavtec_phoneme_model",
        "model_path": "models/wav2vec2-phoneme",
        "duration_seconds": 0.8,
        "frame_count": len(rows),
        "vocabulary": VOCAB,
        "blank_token_id": 0,
        "log_probs": rows,
        "decoded_phonemes": decoded,
    }


def _loader(tmp_path):
    cmu = tmp_path / "cmu"
    cmu.mkdir()
    (cmu / "cmudict.dict").write_text("BOOK B UH K\nBACK B AE K\n", encoding="utf-8")
    (cmu / "cmudict.phones").write_text("B stop\nUH vowel\nK stop\nAE vowel\n", encoding="utf-8")
    (cmu / "cmudict.symbols").write_text("B\nUH\nK\nAE\n", encoding="utf-8")
    return CMUDictLoader(cmu / "cmudict.dict", cmu / "cmudict.phones", cmu / "cmudict.symbols").load()


def test_acoustic_gop_correct_word_high_score() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="cat",
        prompt_type="word",
        raw_transcript="cat",
        acoustic_evidence=_evidence([1, 2, 3]),
        config={"word_threshold": 0.75},
    )

    assert gop["gop_supported"] is True
    assert gop["alignment_quality"] == "usable"
    assert gop["overall_gop_score"] >= 0.75
    assert [item["phoneme"] for item in gop["phoneme_scores"]] == ["K", "AE", "T"]


def test_acoustic_gop_maps_arpabet_expected_phonemes_to_ipa_model_vocabulary() -> None:
    ipa_vocab = {
        0: "<pad>",
        1: "l",
        2: "ɔ",
        3: "ɡ",
        4: "b",
    }

    def row(best_id: int) -> list[float]:
        probs = [0.02] * len(ipa_vocab)
        probs[0] = 0.03
        probs[best_id] = 0.90
        total = sum(probs)
        return [math.log(value / total) for value in probs]

    evidence = {
        "available": True,
        "model_version": "existing_wavtec_phoneme_model",
        "model_path": "models/wav2vec2-phoneme",
        "duration_seconds": 0.8,
        "frame_count": 7,
        "vocabulary": ipa_vocab,
        "blank_token_id": 0,
        "log_probs": [row(0), row(1), row(1), row(2), row(2), row(3), row(0)],
        "decoded_phonemes": ["l", "ɔ", "ɡ"],
    }

    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="log",
        prompt_type="word",
        raw_transcript="log",
        acoustic_evidence=evidence,
        config={"word_threshold": 0.75},
    )

    assert gop["gop_supported"] is True
    assert gop["alignment_quality"] == "usable"
    assert gop["gop_error"] is None
    assert [item["phoneme"] for item in gop["phoneme_scores"]] == ["L", "AO", "G"]
    assert [item["model_phoneme"] for item in gop["phoneme_scores"]] == ["l", "ɔ", "ɡ"]


def test_short_word_gop_assist_accepts_strong_consonants_with_allowed_vowel_variant(tmp_path) -> None:
    ipa_vocab = {0: "<pad>", 1: "b", 2: "\u028a", 3: "k", 4: "u\u02d0"}

    def row(best_id: int, weak_expected_id: int | None = None, competitor_id: int | None = None) -> list[float]:
        probs = [0.01] * len(ipa_vocab)
        probs[0] = 0.03
        probs[best_id] = 0.90
        if weak_expected_id is not None and competitor_id is not None:
            probs[best_id] = 0.03
            probs[weak_expected_id] = 0.08
            probs[competitor_id] = 0.78
        total = sum(probs)
        return [math.log(value / total) for value in probs]

    loader = _loader(tmp_path)
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="book",
        prompt_type="word",
        raw_transcript="look",
        acoustic_evidence={
            "available": True,
            "model_version": "existing_wavtec_phoneme_model",
            "model_path": "models/wav2vec2-phoneme",
            "duration_seconds": 0.8,
            "frame_count": 8,
            "vocabulary": ipa_vocab,
            "blank_token_id": 0,
            "log_probs": [row(0), row(1), row(1), row(2, 2, 4), row(2, 2, 4), row(3), row(3), row(0)],
            "decoded_phonemes": ["b", "u\u02d0", "k"],
        },
        cmudict_loader=loader,
        config={"word_threshold": 0.75, "short_word_assist_min_overall": 0.60},
    )
    normalization = normalize_asr_transcript(
        raw_transcript="look",
        expected_text="book",
        prompt_type="word",
        observed_phonemes=["L", "UH", "K"],
        cmudict_loader=loader,
    )
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert gop["gop_decision"] == "accepted_by_short_word_gop_structure"
    assert gop["gop_short_word_assist"]["accepted"] is True
    assert updated["accepted"] is True
    assert updated["raw_transcript"] == "look"
    assert updated["corrected_transcript"] == "book"
    assert updated["correction_strategy_used"] == "gop_assisted_short_word_structure"


def test_short_word_gop_assist_rejects_unsupported_vowel_competitor(tmp_path) -> None:
    ipa_vocab = {0: "<pad>", 1: "b", 2: "\u028a", 3: "k", 4: "\u00e6"}

    def row(best_id: int, weak_expected_id: int | None = None, competitor_id: int | None = None) -> list[float]:
        probs = [0.01] * len(ipa_vocab)
        probs[0] = 0.03
        probs[best_id] = 0.90
        if weak_expected_id is not None and competitor_id is not None:
            probs[best_id] = 0.03
            probs[weak_expected_id] = 0.08
            probs[competitor_id] = 0.78
        total = sum(probs)
        return [math.log(value / total) for value in probs]

    loader = _loader(tmp_path)
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="book",
        prompt_type="word",
        raw_transcript="back",
        acoustic_evidence={
            "available": True,
            "model_version": "existing_wavtec_phoneme_model",
            "model_path": "models/wav2vec2-phoneme",
            "duration_seconds": 0.8,
            "frame_count": 8,
            "vocabulary": ipa_vocab,
            "blank_token_id": 0,
            "log_probs": [row(0), row(1), row(1), row(2, 2, 4), row(2, 2, 4), row(3), row(3), row(0)],
            "decoded_phonemes": ["b", "\u00e6", "k"],
        },
        cmudict_loader=loader,
        config={"word_threshold": 0.75, "short_word_assist_min_overall": 0.60},
    )
    normalization = normalize_asr_transcript(
        raw_transcript="back",
        expected_text="book",
        prompt_type="word",
        observed_phonemes=["B", "AE", "K"],
        cmudict_loader=loader,
    )
    updated = apply_gop_to_transcript_meta(normalization.to_dict(), gop)

    assert gop["gop_short_word_assist"]["accepted"] is False
    assert updated["accepted"] is False


def test_acoustic_gop_detects_vowel_confusion_log_lug() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="log",
        prompt_type="word",
        raw_transcript="lug",
        acoustic_evidence=_evidence([7, 13, 9], weak_index=1, competitor_id=10),
        config={"word_threshold": 0.75},
    )
    analysis = AIAnalysisService(asr_provider=MockASR())._apply_gop_to_analysis({"is_correct": False, "is_accepted": False, "error_type": ""}, gop)

    assert gop["weak_phoneme"] == "AO"
    assert gop["nearest_confusion"] == "UW"
    assert analysis["error_type"] == "vowel_confusion"


def test_acoustic_gop_detects_initial_consonant_confusion_bat_pat() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="bat",
        prompt_type="word",
        raw_transcript="pat",
        acoustic_evidence=_evidence([6, 2, 3], weak_index=0, competitor_id=5),
        config={"word_threshold": 0.75},
    )
    analysis = AIAnalysisService(asr_provider=MockASR())._apply_gop_to_analysis({"is_correct": False, "is_accepted": False, "error_type": ""}, gop)

    assert gop["weak_phoneme"] == "B"
    assert gop["nearest_confusion"] == "P"
    assert analysis["error_type"] == "initial_sound_substitution"


def test_acoustic_gop_detects_final_sound_omission() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="cat",
        prompt_type="word",
        raw_transcript="ca",
        acoustic_evidence={**_evidence([1, 2, 3], weak_index=2, competitor_id=4), "decoded_phonemes": ["K", "AE"]},
        config={"word_threshold": 0.75},
    )
    analysis = AIAnalysisService(asr_provider=MockASR())._apply_gop_to_analysis({"is_correct": False, "is_accepted": False, "error_type": ""}, gop)

    assert gop["weak_phoneme"] == "T"
    assert analysis["error_type"] == "final_sound_omission"


def test_gop_disabled_fallback() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="cat",
        prompt_type="word",
        raw_transcript="cat",
        acoustic_evidence=_evidence([1, 2, 3]),
        config={"enabled": False},
    )

    assert gop["gop_enabled"] is False
    assert gop["gop_supported"] is False
    assert gop["gop_decision"] == "disabled"


def test_missing_phoneme_mapping_fails_safely() -> None:
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="ship",
        prompt_type="word",
        raw_transcript="ship",
        acoustic_evidence=_evidence([1, 2, 3]),
        config={"word_threshold": 0.75},
    )

    assert gop["gop_supported"] is False
    assert gop["alignment_quality"] == "failed"
    assert "not found" in (gop["gop_error"] or "")


def test_alignment_failure_fallback_for_empty_frames() -> None:
    alignment = ctc_forced_align(expected_phonemes=["K"], frame_log_probs=[], vocabulary=VOCAB, blank_token_id=0)

    assert alignment["ok"] is False
    assert "empty" in alignment["error"]


def test_letter_sound_mapping_differs_from_letter_name_mapping() -> None:
    sound, sound_source, _ = canonical_expected_phonemes("H", prompt_type="letter", task_type="letter_sound")
    name, name_source, _ = canonical_expected_phonemes("H", prompt_type="letter", task_type="crla_task_1_letter")

    assert sound == ["HH"]
    assert sound_source == "letter_sound_map"
    assert name != sound
    assert name_source != "letter_sound_map"


def test_expected_centric_layer_does_not_accept_on_gop_alone() -> None:
    normalization = normalize_asr_transcript(
        raw_transcript="banana",
        expected_text="cat",
        prompt_type="word",
        observed_phonemes=["B", "AH"],
    )
    gop = compute_gop(
        audio_path_or_waveform=None,
        expected_text="cat",
        prompt_type="word",
        raw_transcript="banana",
        acoustic_evidence=_evidence([1, 2, 3]),
        config={"word_threshold": 0.75},
    )

    updated = AIAnalysisService(asr_provider=MockASR())
    transcript_meta = updated._apply_gop_to_analysis({"is_correct": False, "is_accepted": False, "error_type": ""}, gop)

    assert normalization.accepted is False
    assert transcript_meta["error_type"] != "correct"
