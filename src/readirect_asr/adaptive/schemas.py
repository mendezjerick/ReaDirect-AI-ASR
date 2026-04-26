from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LearnerResponseHistoryItem(BaseModel):
    prompt_id: str | None = None
    module_key: str | None = None
    activity_type: str | None = None
    expected_text: str | None = None
    actual_text: str | None = None
    is_correct: bool | None = None
    similarity_label: str | None = None
    error_type: str | None = None
    skill_signal: str | None = None
    target_phoneme: str | None = None
    target_position: str | None = None
    difficulty_level: str | None = None
    difficulty_score: float | None = None
    timestamp: str | None = None


class LearnerStateSummary(BaseModel):
    total_attempts: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    accuracy: float = 0.0
    recent_accuracy: float = 0.0
    correct_streak: int = 0
    incorrect_streak: int = 0
    error_type_counts: dict[str, int] = Field(default_factory=dict)
    skill_signal_counts: dict[str, int] = Field(default_factory=dict)
    target_phoneme_counts: dict[str, int] = Field(default_factory=dict)
    weak_skill_signals: list[str] = Field(default_factory=list)
    strong_skill_signals: list[str] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)
    difficulty_adjustment: str = "same"
    notes: list[str] = Field(default_factory=list)


class CandidateItem(BaseModel):
    prompt_id: str | None = None
    module_key: str | None = None
    task_type: str | None = None
    activity_type: str | None = None
    prompt_text: str | None = None
    expected_text: str | None = None
    accepted_answers: Any = None
    expected_phonemes: Any = None
    initial_phoneme: str | None = None
    vowel_phonemes: Any = None
    final_phoneme: str | None = None
    phoneme_pattern: str | None = None
    skill_tag: str | None = None
    skill_group: str | None = None
    error_focus: str | None = None
    target_position: str | None = None
    target_phoneme: str | None = None
    difficulty_level: str | None = None
    difficulty_score: float | None = None
    adaptive_bucket: str | None = None
    recommended_for_error_type: str | None = None
    practice_role: str | None = None
    mastery_candidate: bool | None = None
    review_candidate: bool | None = None
    is_active: bool | None = True
    needs_manual_review: bool | None = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdaptiveRecommendation(BaseModel):
    selected_item: dict[str, Any] | None = None
    ranked_candidates: list[dict[str, Any]] = Field(default_factory=list)
    learner_summary: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    debug_info: dict[str, Any] | None = None
