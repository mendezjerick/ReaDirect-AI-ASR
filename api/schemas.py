from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    asr_provider: str
    content_index_loaded: bool
    cmudict_loaded: bool
    service_status: str = "ok"
    asr_architecture: str = "wav2vec2_only"
    active_asr_model: str = "wav2vec2"
    wav2vec2_asr_available: bool = False
    wav2vec2_asr_model_name: str = ""
    active_asr_model_path: str = ""
    wav2vec2_phoneme_available: bool = False
    wav2vec2_phoneme_model_name: str = ""
    whisper_available: bool = False
    whisper_removed: bool = True
    model_version: str = ""
    base_model: str = ""
    training_type: str = ""
    training_mix: str = ""
    supported_prompt_types: list[str] = Field(default_factory=lambda: ["letter", "word", "sentence"])
    correction_layer_enabled: bool = True
    expected_centric_scoring_enabled: bool = True
    phoneme_evidence_enabled: bool = True
    thresholds: dict[str, Any] = Field(default_factory=dict)
    local_model_paths_loaded: bool = False
    missing_model_paths: list[str] = Field(default_factory=list)
    reinforcement_corrections_enabled: bool = True
    reinforcement_corrections_dir: str = "reinforcement-learning"
    reinforcement_files_loaded: list[str] = Field(default_factory=list)
    reinforcement_letter_rules_count: int = 0
    reinforcement_word_rules_count: int = 0
    reinforcement_load_warnings: list[str] = Field(default_factory=list)
    audio_quality_validation_enabled: bool = True
    audio_quality_thresholds: dict[str, Any] = Field(default_factory=dict)
    pause_detection_enabled: bool = True
    uncertainty_decision_enabled: bool = True


class VersionResponse(BaseModel):
    service: str
    version: str
    asr_provider: str
    config: dict[str, Any]


class AnalyzeTextRequest(BaseModel):
    expected_text: str | None = None
    actual_text: str
    accepted_answers: list[str] = Field(default_factory=list)
    prompt_id: str | None = None
    module_key: str | None = None
    activity_type: str | None = None
    task_type: str | None = None
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    learner_history: list[dict[str, Any]] = Field(default_factory=list)
    candidate_items: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False


class AnalyzeAudioRequest(BaseModel):
    audio_path: str | None = None
    expected_text: str | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    prompt_id: str | None = None
    prompt_type: str | None = None
    module_key: str | None = None
    activity_type: str | None = None
    task_type: str | None = None
    learner_response_id: str | int | None = None
    attempt_id: str | int | None = None
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    learner_history: list[dict[str, Any]] = Field(default_factory=list)
    candidate_items: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False
    developer_reinforcement_enabled: bool = False
    developer_user_role: str | None = None
    developer_user_id: str | int | None = None


class ReinforcementCorrectionRequest(BaseModel):
    expected_text: str
    raw_transcript: str
    prompt_type: str
    accepted: bool = False
    retry_required: bool = False
    uncertain: bool = False
    correction_strategy_used: str = "none"
    created_by: str = "admin"
    source: str = "true_sandbox_supervised"
    notes: str = "manually approved from True Sandbox"
    supervised_reinforcement_enabled: bool = True
    developer_reinforcement_enabled: bool = True
    developer_user_role: str | None = None


class ReinforcementCorrectionResponse(BaseModel):
    saved: bool
    target_file: str = ""
    reason: str
    duplicate: bool = False


class AnalysisResponse(BaseModel):
    ok: bool
    request_id: str
    mode: str
    provider: str
    model_size: Optional[str] = None
    prompt_id: Optional[str] = None
    expected_text: str = ""
    accepted_answers: list[str] = Field(default_factory=list)
    transcript: str = ""
    normalized_transcript: str = ""
    raw_transcript: str = ""
    corrected_transcript: str = ""
    displayed_transcript: str = ""
    prompt_type: str = "unknown"
    asr_route: str = "wav2vec2_only"
    model_family: str = "wav2vec2"
    model_used: str = ""
    wav2vec2_transcript: str = ""
    whisper_transcript: Optional[str] = None
    whisper_removed: bool = True
    raw_wer: float = 0.0
    corrected_wer: float = 0.0
    raw_cer: float = 0.0
    corrected_cer: float = 0.0
    phonetic_similarity_score: float = 0.0
    composite_score: float = 0.0
    accepted: bool = False
    normalization_applied: bool = False
    normalization_reason: str = ""
    correction_strategy_used: str = "none"
    accepted_by_letter_alias: bool = False
    accepted_by_phonetic_threshold: bool = False
    accepted_by_known_confusion: bool = False
    accepted_by_letter_lattice: bool = False
    accepted_by_letter_normalization: bool = False
    accepted_by_exact_match: bool = False
    accepted_by_vowel_tail: bool = False
    accepted_by_phoneme_evidence: bool = False
    gop_enabled: bool = True
    gop_available: bool = False
    gop_score: Optional[float] = None
    gop_confidence: Optional[float] = None
    gop_decision: str = "not_available"
    gop_threshold: Optional[float] = None
    gop_prompt_type: str = "unknown"
    gop_expected_phonemes: list[str] = Field(default_factory=list)
    gop_observed_phonemes: list[str] = Field(default_factory=list)
    gop_phoneme_scores: list[dict[str, Any]] = Field(default_factory=list)
    gop_word_scores: list[dict[str, Any]] = Field(default_factory=list)
    mispronounced_phonemes: list[str] = Field(default_factory=list)
    weak_words: list[str] = Field(default_factory=list)
    gop_correction_applied: bool = False
    gop_error: Optional[str] = None
    dynamic_correction_enabled: bool = True
    dynamic_correction_applied: bool = False
    dynamic_correction_strategy: str = "dynamic_expected_word_correction"
    dynamic_correction_sub_strategy: str = ""
    dynamic_correction_confidence: Optional[float] = None
    dynamic_correction_threshold: Optional[float] = None
    dynamic_spelling_similarity: Optional[float] = None
    dynamic_phoneme_similarity: Optional[float] = None
    dynamic_gop_score: Optional[float] = None
    dynamic_homophone_match: bool = False
    dynamic_context_score: Optional[float] = None
    dynamic_correction_reason: str = ""
    dynamic_suspicious_fragment: bool = False
    dynamic_fragment_reasons: list[str] = Field(default_factory=list)
    dynamic_phoneme_coverage: Optional[float] = None
    asr_spelling_variant_enabled: bool = True
    asr_spelling_variant_applied: bool = False
    asr_spelling_variant_strategy: str = "dynamic_asr_spelling_variant"
    asr_spelling_variant_sub_strategy: str = ""
    asr_spelling_variant_confidence: Optional[float] = None
    asr_spelling_variant_threshold: Optional[float] = None
    consonant_skeleton_similarity: Optional[float] = None
    vowel_tolerant_similarity: Optional[float] = None
    expected_phoneme_coverage: Optional[float] = None
    variant_edit_similarity: Optional[float] = None
    variant_reason: str = ""
    word_alignment: list[dict[str, Any]] = Field(default_factory=list)
    accepted_by_reinforcement_match: bool = False
    reinforcement_source_file: str = ""
    reinforcement_expected_label: str = ""
    reinforcement_matched_transcript: str = ""
    reinforcement_match_normalized: dict[str, Any] = Field(default_factory=dict)
    reinforcement_match_original: dict[str, Any] = Field(default_factory=dict)
    critical_phoneme: Optional[str] = None
    critical_phoneme_detected: Optional[bool] = None
    critical_phoneme_expected_position: Optional[str] = None
    critical_phoneme_reason: Optional[str] = None
    critical_pair_detected: bool = False
    confidence_level: str = ""
    threshold_used: float = 0.0
    confidence_or_threshold_used: float = 0.0
    confidence: Optional[float] = None
    is_correct: bool = False
    is_exact: bool = False
    is_accepted: bool = False
    character_similarity: float = 0.0
    token_similarity: float = 0.0
    similarity_label: str = "blank"
    expected_phonemes: list[str] = Field(default_factory=list)
    expected_phoneme_source: str = ""
    expected_phoneme_variants: list[list[str]] = Field(default_factory=list)
    observed_phonemes: list[str] = Field(default_factory=list)
    actual_phonemes: list[str] = Field(default_factory=list)
    phoneme_similarity: float = 0.0
    error_type: str = ""
    error_position: Optional[str] = None
    feedback_hint: str = ""
    coach_hint_key: str = ""
    learner_safe_summary: str = ""
    skill_signal: str = ""
    target_phoneme: str = ""
    target_position: str = ""
    recommended_practice_focus: str = ""
    recommended_action: str = ""
    adaptive_recommendation: Optional[dict[str, Any]] = None
    learner_summary: Optional[dict[str, Any]] = None
    audio_quality: dict[str, Any] = Field(default_factory=dict)
    pause_metrics: dict[str, Any] = Field(default_factory=dict)
    uncertain: bool = False
    retry_required: bool = False
    uncertainty_reasons: list[str] = Field(default_factory=list)
    quality_gate_failed: bool = False
    learner_retry_message: str = ""
    developer_quality_notes: list[str] = Field(default_factory=list)
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    enrichment_metadata: dict[str, Any] = Field(default_factory=dict)
    analysis_source: str = "heuristic_transcript_phoneme"
    debug_metadata: dict[str, Any] = Field(default_factory=dict)
    developer_reinforcement_mode: bool = False
    reinforcement_saved: bool = False
    reinforcement_duplicate: bool = False
    reinforcement_target_file: str = ""
    reinforcement_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    debug_info: Optional[dict[str, Any]] = None
    processing_seconds: float = 0.0
    error: Optional[str] = None


class ContentItemRequest(BaseModel):
    prompt_id: str | None = None
    expected_text: str | None = None
    module_key: str | None = None
    activity_type: str | None = None
    task_type: str | None = None


class ContentItemResponse(BaseModel):
    ok: bool
    prompt_id: str | None = None
    found: bool
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    enrichment_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RecommendNextRequest(BaseModel):
    learner_history: list[dict[str, Any]] = Field(default_factory=list)
    current_context: dict[str, Any] = Field(default_factory=dict)
    candidate_items: list[dict[str, Any]] = Field(default_factory=list)
    module_key: str | None = None
    activity_type: str | None = None
    top_k: int = 5
    debug: bool = False


class RecommendNextResponse(BaseModel):
    ok: bool
    selected_item: Optional[dict[str, Any]] = None
    ranked_candidates: list[dict[str, Any]] = Field(default_factory=list)
    learner_summary: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    debug_info: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# Backward-compatible aliases for earlier Phase 6 names.
AnalyzeAudioResponse = AnalysisResponse
AnalyzeTextResponse = AnalysisResponse
AnalyzeContentItemRequest = ContentItemRequest
AnalyzeContentItemResponse = ContentItemResponse
