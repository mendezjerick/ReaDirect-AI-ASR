from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from api.errors import new_request_id
from api.schemas import (
    AnalysisResponse,
    AnalyzeAudioRequest,
    AnalyzeTextRequest,
    ContentItemRequest,
    ContentItemResponse,
    RecommendNextRequest,
    RecommendNextResponse,
)
from readirect_asr.adaptive.recommendation import AdaptiveRecommendationEngine
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.result import ASRResult
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.content.enricher import ContentEnricher
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.answer_matching import parse_accepted_answers
from readirect_asr.scoring.reading_analyzer import analyze_reading_response
from readirect_asr.text.normalization import normalize_transcript

logger = logging.getLogger("readirect_ai_asr")


class AIAnalysisService:
    def __init__(
        self,
        asr_provider: Any | None = None,
        cmudict_loader: CMUDictLoader | None = None,
        content_repository: ContentRepository | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {"api": {"debug": True}, "asr": {"provider": "mock"}}
        self.asr_provider = asr_provider or MockASR()
        self.cmudict_loader = cmudict_loader or CMUDictLoader().load()
        self.content_repository = content_repository or ContentRepository().load()
        self.content_enricher = ContentEnricher(self.cmudict_loader)
        self.adaptive_engine = AdaptiveRecommendationEngine(
            content_repository=self.content_repository,
            config=self.config.get("adaptive", {}),
        )

    @property
    def provider_name(self) -> str:
        return str(getattr(self.asr_provider, "provider", self.config.get("asr", {}).get("provider", "mock")))

    @property
    def model_size(self) -> str | None:
        return getattr(self.asr_provider, "model_size", None) or self.config.get("asr", {}).get("model_size")

    def analyze_text(self, request: AnalyzeTextRequest) -> AnalysisResponse:
        started = time.perf_counter()
        request_id = new_request_id()
        warnings: list[str] = []
        try:
            context = self.resolve_expected_context(
                prompt_id=request.prompt_id,
                expected_text=request.expected_text,
                accepted_answers=request.accepted_answers,
                content_metadata={
                    **request.content_metadata,
                    "module_key": request.module_key or request.content_metadata.get("module_key", ""),
                    "activity_type": request.activity_type or request.content_metadata.get("activity_type", ""),
                    "task_type": request.task_type or request.content_metadata.get("task_type", ""),
                },
                warnings=warnings,
            )
            expected_text = str(context["expected_text"])
            if not expected_text:
                return self._error_response("text", request_id, "missing_expected_text", "Expected text was not provided and could not be resolved.", started, warnings)
            analysis = self.run_reading_analysis(expected_text, request.actual_text, context["accepted_answers"], context["content_metadata"])
            adaptive = self._maybe_recommend(request.learner_history, request.candidate_items, context, analysis, request.debug)
            return self.build_response(
                request_id=request_id,
                mode="text",
                prompt_id=context.get("prompt_id") or request.prompt_id,
                expected_text=expected_text,
                accepted_answers=context["accepted_answers"],
                transcript=request.actual_text,
                normalized_transcript=normalize_transcript(request.actual_text),
                analysis=analysis,
                content_metadata=context["content_metadata"],
                enrichment_metadata=context["enrichment_metadata"],
                warnings=warnings,
                started=started,
                debug=request.debug,
                adaptive=adaptive,
            )
        except Exception as exc:
            return self._error_response("text", request_id, "analysis_failed", "Text analysis failed.", started, warnings, exc)

    def analyze_audio(self, request: AnalyzeAudioRequest) -> AnalysisResponse:
        started = time.perf_counter()
        request_id = new_request_id()
        warnings: list[str] = []
        try:
            context = self.resolve_expected_context(
                prompt_id=request.prompt_id,
                expected_text=request.expected_text,
                accepted_answers=request.accepted_answers,
                content_metadata={
                    **request.content_metadata,
                    "module_key": request.module_key or request.content_metadata.get("module_key", ""),
                    "activity_type": request.activity_type or request.content_metadata.get("activity_type", ""),
                    "task_type": request.task_type or request.content_metadata.get("task_type", ""),
                },
                warnings=warnings,
            )
            expected_text = str(context["expected_text"])
            if not expected_text:
                return self._error_response("audio", request_id, "missing_expected_text", "Expected text was not provided and could not be resolved.", started, warnings)
            if not request.audio_path:
                return self._error_response("audio", request_id, "missing_audio_path", "Audio path was not provided.", started, warnings)

            audio_path = Path(request.audio_path)
            if not audio_path.exists():
                return self._error_response("audio", request_id, "audio_file_not_found", "Audio file was not found.", started, warnings, debug_info={"audio_path": str(audio_path)} if request.debug else None)

            asr = self.transcribe_audio(str(audio_path), expected_text=expected_text, content_metadata=context["content_metadata"])
            if asr.error:
                warnings.append("transcription_failed")
                return self._error_response("audio", request_id, "transcription_failed", asr.error, started, warnings, debug_info=asr.to_dict() if request.debug else None)

            analysis = self.run_reading_analysis(expected_text, asr.transcript, context["accepted_answers"], context["content_metadata"])
            adaptive = self._maybe_recommend(request.learner_history, request.candidate_items, context, analysis, request.debug)
            return self.build_response(
                request_id=request_id,
                mode="audio",
                prompt_id=context.get("prompt_id") or request.prompt_id,
                expected_text=expected_text,
                accepted_answers=context["accepted_answers"],
                transcript=asr.transcript,
                normalized_transcript=asr.normalized_transcript or normalize_transcript(asr.transcript),
                confidence=asr.confidence,
                analysis=analysis,
                content_metadata=context["content_metadata"],
                enrichment_metadata=context["enrichment_metadata"],
                warnings=warnings,
                started=started,
                debug=request.debug,
                debug_info={"asr": asr.to_dict(), "learner_response_id": request.learner_response_id, "attempt_id": request.attempt_id},
                adaptive=adaptive,
            )
        except Exception as exc:
            return self._error_response("audio", request_id, "analysis_failed", "Audio analysis failed.", started, warnings, exc)

    def get_content_item(self, request: ContentItemRequest) -> ContentItemResponse:
        warnings: list[str] = []
        metadata = self.content_repository.get_metadata(request.prompt_id, request.expected_text)
        enrichment = self.content_repository.get_enrichment(request.prompt_id, request.expected_text)
        found = bool(metadata or enrichment)
        if not found and request.expected_text:
            row = {
                "prompt_id": request.prompt_id or "",
                "expected_text": request.expected_text,
                "prompt_text": request.expected_text,
                "module_key": request.module_key or "",
                "activity_type": request.activity_type or "",
                "task_type": request.task_type or "",
            }
            enrichment = self._pick_enrichment(self.content_enricher.enrich_row(row))
            metadata = {key: value for key, value in row.items() if value}
            warnings.append("content_item_not_found_generated_from_request")
        elif not found:
            warnings.append("content_item_not_found")
        return ContentItemResponse(
            ok=found or bool(request.expected_text),
            prompt_id=request.prompt_id or metadata.get("prompt_id"),
            found=found,
            content_metadata=_json_safe(metadata),
            enrichment_metadata=_json_safe(enrichment),
            warnings=warnings,
        )

    def recommend_next(self, request: RecommendNextRequest) -> RecommendNextResponse:
        context = dict(request.current_context or {})
        if request.module_key:
            context["module_key"] = request.module_key
        if request.activity_type:
            context["activity_type"] = request.activity_type
        result = self.adaptive_engine.recommend_next(
            history=request.learner_history,
            current_context=context,
            candidate_items=request.candidate_items,
            top_k=request.top_k,
            debug=request.debug,
        )
        response = self.adaptive_engine.build_recommendation_response(result)
        return RecommendNextResponse(**_json_safe(response))

    def resolve_expected_context(
        self,
        prompt_id: str | None,
        expected_text: str | None,
        accepted_answers: list[str] | None,
        content_metadata: dict[str, Any] | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        repo_metadata = self.content_repository.get_metadata(prompt_id, expected_text)
        repo_enrichment = self.content_repository.get_enrichment(prompt_id, expected_text)
        merged_metadata = {**repo_metadata, **(content_metadata or {})}
        resolved_expected = expected_text or str(merged_metadata.get("expected_text", "") or "")
        accepted = accepted_answers or parse_accepted_answers(merged_metadata.get("accepted_answers", ""))
        if prompt_id and not repo_metadata:
            warnings.append("content_item_not_found")
        if not repo_enrichment and resolved_expected:
            row = {
                **merged_metadata,
                "prompt_id": prompt_id or merged_metadata.get("prompt_id", ""),
                "expected_text": resolved_expected,
                "prompt_text": merged_metadata.get("prompt_text", resolved_expected),
            }
            repo_enrichment = self._pick_enrichment(self.content_enricher.enrich_row(row))
            warnings.append("enrichment_generated_from_request")
        return {
            "prompt_id": prompt_id or merged_metadata.get("prompt_id"),
            "expected_text": resolved_expected,
            "accepted_answers": accepted,
            "content_metadata": _json_safe(merged_metadata),
            "enrichment_metadata": _json_safe(repo_enrichment),
        }

    def transcribe_audio(self, audio_path: str, expected_text: str = "", content_metadata: dict[str, Any] | None = None) -> ASRResult:
        metadata = content_metadata or {}
        if self.provider_name == "mock":
            transcript = str(metadata.get("mock_transcript") or metadata.get("actual_text") or expected_text or "")
            return ASRResult(
                transcript=transcript,
                normalized_transcript=normalize_transcript(transcript),
                provider="mock",
                model_size=None,
            )
        result = self.asr_provider.transcribe(audio_path)
        if isinstance(result, ASRResult):
            return result
        transcript = str(result.get("transcript", ""))
        return ASRResult(
            transcript=transcript,
            normalized_transcript=normalize_transcript(transcript),
            confidence=result.get("confidence"),
            provider=str(result.get("provider", self.provider_name)),
            model_size=self.model_size,
            error=result.get("error"),
        )

    def run_reading_analysis(
        self,
        expected_text: str,
        actual_text: str,
        accepted_answers: list[str],
        content_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return analyze_reading_response(
            expected_text=expected_text,
            actual_text=actual_text,
            accepted_answers=accepted_answers,
            cmudict_loader=self.cmudict_loader,
            content_metadata=content_metadata,
        )

    def build_response(
        self,
        request_id: str,
        mode: str,
        prompt_id: str | None,
        expected_text: str,
        accepted_answers: list[str],
        transcript: str,
        normalized_transcript: str,
        analysis: dict[str, Any],
        content_metadata: dict[str, Any],
        enrichment_metadata: dict[str, Any],
        warnings: list[str],
        started: float,
        debug: bool,
        confidence: float | None = None,
        debug_info: dict[str, Any] | None = None,
        adaptive: dict[str, Any] | None = None,
    ) -> AnalysisResponse:
        include_debug = debug and bool(self.config.get("api", {}).get("debug", True))
        if include_debug and debug_info is None:
            debug_info = {
                "content_index_path": str(self.content_repository.loaded_path) if self.content_repository.loaded_path else "",
                "asr_provider": self.provider_name,
            }
        merged_enrichment = {**enrichment_metadata}
        response = AnalysisResponse(
            ok=True,
            request_id=request_id,
            mode=mode,
            provider=self.provider_name,
            model_size=self.model_size,
            prompt_id=prompt_id,
            expected_text=expected_text,
            accepted_answers=accepted_answers,
            transcript=transcript,
            normalized_transcript=normalized_transcript,
            confidence=confidence,
            is_correct=bool(analysis.get("is_correct", False)),
            is_exact=bool(analysis.get("is_exact", False)),
            is_accepted=bool(analysis.get("is_accepted", False)),
            character_similarity=float(analysis.get("character_similarity", 0.0) or 0.0),
            token_similarity=float(analysis.get("token_similarity", 0.0) or 0.0),
            similarity_label=str(analysis.get("similarity_label", "")),
            expected_phonemes=list(analysis.get("expected_phonemes", []) or []),
            actual_phonemes=list(analysis.get("actual_phonemes", []) or []),
            phoneme_similarity=float(analysis.get("phoneme_similarity", 0.0) or 0.0),
            error_type=str(analysis.get("error_type", "")),
            error_position=analysis.get("error_position"),
            feedback_hint=str(analysis.get("feedback_hint", "")),
            coach_hint_key=str(analysis.get("coach_hint_key", "")),
            learner_safe_summary=str(analysis.get("learner_safe_summary", "")),
            skill_signal=str(analysis.get("skill_signal", "")),
            target_phoneme=str(analysis.get("target_phoneme", "")),
            target_position=str(analysis.get("target_position", "")),
            recommended_practice_focus=str(analysis.get("recommended_practice_focus", "")),
            recommended_action=str(analysis.get("recommended_action", "")),
            adaptive_recommendation=_json_safe(adaptive.get("recommendation")) if adaptive else None,
            learner_summary=_json_safe(adaptive.get("learner_summary")) if adaptive else None,
            content_metadata=_json_safe(content_metadata),
            enrichment_metadata=_json_safe(merged_enrichment),
            analysis_source=str(analysis.get("analysis_source", "heuristic_transcript_phoneme")),
            warnings=warnings,
            debug_info=_json_safe(debug_info) if include_debug else None,
            processing_seconds=round(time.perf_counter() - started, 3),
            error=None,
        )
        logger.info("request_id=%s endpoint=%s provider=%s prompt_id=%s ok=true", request_id, mode, self.provider_name, prompt_id)
        return response

    def _maybe_recommend(
        self,
        learner_history: list[dict[str, Any]],
        candidate_items: list[dict[str, Any]],
        context: dict[str, Any],
        analysis: dict[str, Any],
        debug: bool,
    ) -> dict[str, Any] | None:
        if not learner_history:
            return None
        current_context = {
            **(context.get("content_metadata") or {}),
            **(context.get("enrichment_metadata") or {}),
            "module_key": (context.get("content_metadata") or {}).get("module_key"),
            "activity_type": (context.get("content_metadata") or {}).get("activity_type"),
            "last_error_type": analysis.get("error_type"),
            "last_skill_signal": analysis.get("skill_signal"),
            "target_phoneme": analysis.get("target_phoneme"),
        }
        history = [*learner_history, _analysis_to_history_item(context, analysis)]
        return self.adaptive_engine.recommend_next(
            history=history,
            current_context=current_context,
            candidate_items=candidate_items,
            top_k=int(self.config.get("adaptive", {}).get("top_k_default", 5) or 5),
            debug=debug,
        )

    def _error_response(
        self,
        mode: str,
        request_id: str,
        error: str,
        warning: str,
        started: float,
        warnings: list[str] | None = None,
        exception: Exception | None = None,
        debug_info: dict[str, Any] | None = None,
    ) -> AnalysisResponse:
        active_warnings = [*(warnings or []), warning]
        include_debug = bool(self.config.get("api", {}).get("debug", True))
        if exception and include_debug:
            debug_info = {**(debug_info or {}), "exception": str(exception)}
        logger.warning("request_id=%s endpoint=%s provider=%s ok=false error=%s", request_id, mode, self.provider_name, error)
        return AnalysisResponse(
            ok=False,
            request_id=request_id,
            mode=mode,
            provider=self.provider_name,
            model_size=self.model_size,
            warnings=active_warnings,
            debug_info=_json_safe(debug_info) if debug_info and include_debug else None,
            processing_seconds=round(time.perf_counter() - started, 3),
            error=error,
        )

    def _pick_enrichment(self, enrichment: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "expected_phonemes",
            "initial_phoneme",
            "vowel_phonemes",
            "final_phoneme",
            "phoneme_pattern",
            "skill_tag",
            "skill_group",
            "error_focus",
            "target_position",
            "target_phoneme",
            "difficulty_level",
            "difficulty_score",
            "adaptive_bucket",
            "recommended_for_error_type",
            "practice_role",
            "mastery_candidate",
            "needs_manual_review",
        ]
        return {key: enrichment.get(key) for key in keys if key in enrichment}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _analysis_to_history_item(context: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    metadata = context.get("content_metadata") or {}
    enrichment = context.get("enrichment_metadata") or {}
    return {
        "prompt_id": context.get("prompt_id") or metadata.get("prompt_id"),
        "module_key": metadata.get("module_key"),
        "activity_type": metadata.get("activity_type"),
        "expected_text": context.get("expected_text"),
        "actual_text": analysis.get("actual_text"),
        "is_correct": analysis.get("is_correct"),
        "similarity_label": analysis.get("similarity_label"),
        "error_type": analysis.get("error_type"),
        "skill_signal": analysis.get("skill_signal"),
        "target_phoneme": analysis.get("target_phoneme"),
        "target_position": analysis.get("target_position"),
        "difficulty_level": enrichment.get("difficulty_level"),
        "difficulty_score": enrichment.get("difficulty_score"),
    }


# Backward-compatible name used by earlier tests/imports.
AnalysisService = AIAnalysisService
