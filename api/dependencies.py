from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from api.service import AIAnalysisService
from readirect_asr.asr.provider_factory import create_asr_provider
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    config = _load_yaml("configs/service_config.yaml")
    config.setdefault("api", {})
    config.setdefault("analysis", {})
    config.setdefault("asr", {})
    config.setdefault("transcript_normalization", {})
    config.setdefault("audio_quality", {})
    config.setdefault("gop", {})
    config.setdefault("dynamic_expected_correction", {})
    adaptive_config_path = os.getenv("ADAPTIVE_CONFIG_PATH", "configs/adaptive_config.yaml")
    config["adaptive"] = {**_load_yaml(adaptive_config_path), **config.get("adaptive", {})}
    config["api"]["debug"] = os.getenv("API_DEBUG", str(config["api"].get("debug", True))).lower() in {"1", "true", "yes"}
    config["api"]["auth_enabled"] = os.getenv("API_AUTH_ENABLED", str(config["api"].get("auth_enabled", False))).lower() in {"1", "true", "yes"}
    origins = os.getenv("CORS_ALLOW_ORIGINS")
    if origins:
        config["api"]["cors_allow_origins"] = [origin.strip() for origin in origins.split(",") if origin.strip()]
    config["analysis"]["content_index_path"] = os.getenv("CONTENT_INDEX_PATH", config["analysis"].get("content_index_path", "data/manifests/content_index.csv"))
    config["analysis"]["enriched_content_index_path"] = os.getenv("ENRICHED_CONTENT_INDEX_PATH", config["analysis"].get("enriched_content_index_path", "content_bank_enriched/enriched_content_index.csv"))
    config["asr"]["provider"] = os.getenv("ASR_PROVIDER", config["asr"].get("provider", "wav2vec2_only"))
    config["asr"]["model_size"] = os.getenv("ASR_MODEL_SIZE", config["asr"].get("model_size", "models/wav2vec2-readirect-asr-letters-v2"))
    config["asr"]["wav2vec2_asr_model_path"] = os.getenv("WAV2VEC2_ASR_MODEL_PATH", config["asr"].get("wav2vec2_asr_model_path", "models/wav2vec2-readirect-asr-letters-v2"))
    config["asr"]["wav2vec2_phoneme_model_path"] = os.getenv("WAV2VEC2_PHONEME_MODEL_PATH", config["asr"].get("wav2vec2_phoneme_model_path", "models/wav2vec2-phoneme"))
    config["asr"]["wav2vec2_base_asr_model_path"] = os.getenv("WAV2VEC2_BASE_ASR_MODEL_PATH", config["asr"].get("wav2vec2_base_asr_model_path", "models/wav2vec2-readirect-asr"))
    config["asr"]["allow_wav2vec2_base_fallback"] = os.getenv("ALLOW_WAV2VEC2_BASE_FALLBACK", str(config["asr"].get("allow_wav2vec2_base_fallback", False))).lower() in {"1", "true", "yes"}
    config["asr"]["device"] = os.getenv("ASR_DEVICE", config["asr"].get("device", "cpu"))
    config["asr"]["compute_type"] = os.getenv("ASR_COMPUTE_TYPE", config["asr"].get("compute_type", "int8"))
    config["asr"]["use_fp16"] = os.getenv("ASR_USE_FP16", str(config["asr"].get("use_fp16", False))).lower() in {"1", "true", "yes"}
    config["asr"]["language"] = os.getenv("ASR_LANGUAGE", config["asr"].get("language", "en"))
    config["asr"]["task"] = os.getenv("ASR_TASK", config["asr"].get("task", "transcribe"))
    config["asr"]["beam_size"] = int(os.getenv("ASR_BEAM_SIZE", str(config["asr"].get("beam_size", 1))))
    audio_quality = config["audio_quality"]
    audio_quality["min_duration_seconds"] = float(os.getenv("AUDIO_MIN_DURATION_SECONDS", str(audio_quality.get("min_duration_seconds", 1.0))))
    audio_quality["max_duration_seconds"] = float(os.getenv("AUDIO_MAX_DURATION_SECONDS", str(audio_quality.get("max_duration_seconds", 30.0))))
    audio_quality["low_volume_dbfs"] = float(os.getenv("AUDIO_LOW_VOLUME_DBFS", str(audio_quality.get("low_volume_dbfs", -35.0))))
    audio_quality["silence_dbfs"] = float(os.getenv("AUDIO_SILENCE_DBFS", str(audio_quality.get("silence_dbfs", -40.0))))
    audio_quality["mostly_silent_ratio"] = float(os.getenv("AUDIO_MOSTLY_SILENT_RATIO", str(audio_quality.get("mostly_silent_ratio", 0.85))))
    audio_quality["clipping_threshold"] = float(os.getenv("AUDIO_CLIPPING_THRESHOLD", str(audio_quality.get("clipping_threshold", 0.98))))
    audio_quality["clipped_ratio_threshold"] = float(os.getenv("AUDIO_CLIPPED_RATIO_THRESHOLD", str(audio_quality.get("clipped_ratio_threshold", 0.01))))
    audio_quality["min_speech_ratio"] = float(os.getenv("AUDIO_MIN_SPEECH_RATIO", str(audio_quality.get("min_speech_ratio", 0.15))))
    audio_quality["long_pause_seconds"] = float(os.getenv("AUDIO_LONG_PAUSE_SECONDS", str(audio_quality.get("long_pause_seconds", 1.0))))
    audio_quality["very_long_pause_seconds"] = float(os.getenv("AUDIO_VERY_LONG_PAUSE_SECONDS", str(audio_quality.get("very_long_pause_seconds", 2.0))))
    audio_quality["enable_quality_gate"] = _env_bool("AUDIO_ENABLE_QUALITY_GATE", bool(audio_quality.get("enable_quality_gate", True)))
    audio_quality["retry_on_bad_quality"] = _env_bool("AUDIO_RETRY_ON_BAD_QUALITY", bool(audio_quality.get("retry_on_bad_quality", True)))
    normalization = config["transcript_normalization"]
    normalization["phonetic_accept_threshold"] = float(os.getenv("PHONETIC_ACCEPT_THRESHOLD", str(normalization.get("phonetic_accept_threshold", normalization.get("high_similarity_threshold", 0.88)))))
    normalization["phonetic_strict_word_threshold"] = float(os.getenv("PHONETIC_STRICT_WORD_THRESHOLD", str(normalization.get("phonetic_strict_word_threshold", normalization.get("strict_word_threshold", 0.90)))))
    normalization["phonetic_single_letter_threshold"] = float(os.getenv("PHONETIC_SINGLE_LETTER_THRESHOLD", str(normalization.get("phonetic_single_letter_threshold", normalization.get("single_letter_threshold", 0.85)))))
    normalization["phonetic_known_confusion_threshold"] = float(os.getenv("PHONETIC_KNOWN_CONFUSION_THRESHOLD", str(normalization.get("phonetic_known_confusion_threshold", normalization.get("known_confusion_threshold", 0.82)))))
    normalization["phonetic_lattice_threshold"] = float(os.getenv("PHONETIC_LATTICE_THRESHOLD", str(normalization.get("phonetic_lattice_threshold", normalization.get("lattice_threshold", 0.85)))))
    normalization["critical_phoneme_required"] = os.getenv("CRITICAL_PHONEME_REQUIRED", str(normalization.get("critical_phoneme_required", True))).lower() in {"1", "true", "yes"}
    normalization["low_confidence_threshold"] = float(os.getenv("TRANSCRIPT_NORMALIZATION_LOW_CONFIDENCE_THRESHOLD", str(normalization.get("low_confidence_threshold", 0.50))))
    normalization["low_confidence_similarity_threshold"] = float(os.getenv("TRANSCRIPT_NORMALIZATION_LOW_CONFIDENCE_SIMILARITY_THRESHOLD", str(normalization.get("low_confidence_similarity_threshold", 0.95))))
    normalization["reinforcement_corrections_enabled"] = os.getenv("REINFORCEMENT_CORRECTIONS_ENABLED", str(normalization.get("reinforcement_corrections_enabled", True))).lower() in {"1", "true", "yes"}
    normalization["reinforcement_corrections_dir"] = os.getenv("REINFORCEMENT_CORRECTIONS_DIR", str(normalization.get("reinforcement_corrections_dir", "reinforcement-learning")))
    normalization["letter_reinforcement_file"] = os.getenv("LETTER_REINFORCEMENT_FILE", str(normalization.get("letter_reinforcement_file", "letter-reinforcement.csv")))
    normalization["word_reinforcement_file"] = os.getenv("WORD_REINFORCEMENT_FILE", str(normalization.get("word_reinforcement_file", "word-reinforcement.csv")))
    gop = config["gop"]
    gop["enabled"] = _env_bool("GOP_ENABLED", bool(gop.get("enabled", True)))
    gop["letter_threshold"] = float(os.getenv("GOP_LETTER_THRESHOLD", str(gop.get("letter_threshold", 0.70))))
    gop["word_threshold"] = float(os.getenv("GOP_WORD_THRESHOLD", str(gop.get("word_threshold", 0.75))))
    gop["rhyme_threshold"] = float(os.getenv("GOP_RHYME_THRESHOLD", str(gop.get("rhyme_threshold", 0.75))))
    gop["sentence_word_threshold"] = float(os.getenv("GOP_SENTENCE_WORD_THRESHOLD", str(gop.get("sentence_word_threshold", 0.70))))
    gop["passage_word_threshold"] = float(os.getenv("GOP_PASSAGE_WORD_THRESHOLD", str(gop.get("passage_word_threshold", 0.70))))
    gop["min_audio_quality_required"] = _env_bool("GOP_MIN_AUDIO_QUALITY_REQUIRED", bool(gop.get("min_audio_quality_required", True)))
    gop["skip_on_retry_required"] = _env_bool("GOP_SKIP_ON_RETRY_REQUIRED", bool(gop.get("skip_on_retry_required", True)))
    gop["skip_on_uncertain_audio"] = _env_bool("GOP_SKIP_ON_UNCERTAIN_AUDIO", bool(gop.get("skip_on_uncertain_audio", True)))
    gop["debug"] = _env_bool("GOP_DEBUG", bool(gop.get("debug", False)))
    dynamic = config["dynamic_expected_correction"]
    dynamic["enabled"] = _env_bool("DYNAMIC_EXPECTED_CORRECTION_ENABLED", bool(dynamic.get("enabled", True)))
    dynamic["letter_accept_threshold"] = float(os.getenv("DYNAMIC_LETTER_ACCEPT_THRESHOLD", str(dynamic.get("letter_accept_threshold", 0.72))))
    dynamic["word_accept_threshold"] = float(os.getenv("DYNAMIC_WORD_ACCEPT_THRESHOLD", str(dynamic.get("word_accept_threshold", 0.78))))
    dynamic["rhyme_accept_threshold"] = float(os.getenv("DYNAMIC_RHYME_ACCEPT_THRESHOLD", str(dynamic.get("rhyme_accept_threshold", 0.78))))
    dynamic["sentence_word_accept_threshold"] = float(os.getenv("DYNAMIC_SENTENCE_WORD_ACCEPT_THRESHOLD", str(dynamic.get("sentence_word_accept_threshold", 0.80))))
    dynamic["passage_word_accept_threshold"] = float(os.getenv("DYNAMIC_PASSAGE_WORD_ACCEPT_THRESHOLD", str(dynamic.get("passage_word_accept_threshold", 0.82))))
    dynamic["homophone_threshold"] = float(os.getenv("DYNAMIC_HOMOPHONE_THRESHOLD", str(dynamic.get("homophone_threshold", 0.96))))
    dynamic["min_phoneme_for_low_text_match"] = float(os.getenv("DYNAMIC_MIN_PHONEME_FOR_LOW_TEXT_MATCH", str(dynamic.get("min_phoneme_for_low_text_match", 0.90))))
    dynamic["min_gop_for_acceptance"] = float(os.getenv("DYNAMIC_MIN_GOP_FOR_ACCEPTANCE", str(dynamic.get("min_gop_for_acceptance", 0.75))))
    dynamic["skip_on_retry_required"] = _env_bool("DYNAMIC_SKIP_ON_RETRY_REQUIRED", bool(dynamic.get("skip_on_retry_required", True)))
    dynamic["skip_on_uncertain_audio"] = _env_bool("DYNAMIC_SKIP_ON_UNCERTAIN_AUDIO", bool(dynamic.get("skip_on_uncertain_audio", True)))
    dynamic["debug"] = _env_bool("DYNAMIC_DEBUG", bool(dynamic.get("debug", False)))
    return config


@lru_cache(maxsize=1)
def get_cmudict_loader() -> CMUDictLoader:
    cmudict_dir = Path(os.getenv("CMUDICT_DIR", "external_datasets/cmudict"))
    return CMUDictLoader(
        cmudict_dir / "cmudict.dict",
        cmudict_dir / "cmudict.phones",
        cmudict_dir / "cmudict.symbols",
    ).load()


@lru_cache(maxsize=1)
def get_content_repository() -> ContentRepository:
    config = get_config()
    analysis = config.get("analysis", {})
    return ContentRepository(
        content_index_path=analysis.get("content_index_path", "data/manifests/content_index.csv"),
        enriched_content_index_path=analysis.get("enriched_content_index_path", "content_bank_enriched/enriched_content_index.csv"),
        prefer_enriched_content=bool(analysis.get("prefer_enriched_content", True)),
    ).load()


@lru_cache(maxsize=1)
def get_asr_provider():
    return create_asr_provider(get_config().get("asr", {}))


@lru_cache(maxsize=1)
def get_service() -> AIAnalysisService:
    return AIAnalysisService(
        asr_provider=get_asr_provider(),
        cmudict_loader=get_cmudict_loader(),
        content_repository=get_content_repository(),
        config=get_config(),
    )
