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
    module_key: str | None = None
    activity_type: str | None = None
    task_type: str | None = None
    learner_response_id: str | None = None
    attempt_id: str | None = None
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    learner_history: list[dict[str, Any]] = Field(default_factory=list)
    candidate_items: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False


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
    raw_wer: float = 0.0
    corrected_wer: float = 0.0
    phonetic_similarity_score: float = 0.0
    normalization_applied: bool = False
    normalization_reason: str = ""
    correction_strategy_used: str = "none"
    accepted_by_phonetic_threshold: bool = False
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
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    enrichment_metadata: dict[str, Any] = Field(default_factory=dict)
    analysis_source: str = "heuristic_transcript_phoneme"
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
