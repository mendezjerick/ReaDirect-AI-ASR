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
from readirect_asr.audio.preprocessing import analyze_audio_quality, audio_quality_config, validate_audio_file
from readirect_asr.content.content_repository import ContentRepository
from readirect_asr.content.enricher import ContentEnricher
from readirect_asr.correction.dynamic_expected_word_correction import apply_dynamic_expected_word_correction, dynamic_response_fields
from readirect_asr.evaluation.asr_metrics import compute_cer, compute_wer
from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.pronunciation.gop import apply_gop_to_transcript_meta, compute_gop, gop_response_fields
from readirect_asr.scoring.answer_matching import parse_accepted_answers
from readirect_asr.scoring.reading_analyzer import analyze_reading_response
from readirect_asr.text.normalization import normalize_transcript
from readirect_asr.text.transcript_normalizer import TranscriptNormalizationResult, normalize_asr_transcript

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
                warnings.append("missing_expected_text_using_raw_wav2vec2_transcript")
            if not request.audio_path:
                return self._error_response("audio", request_id, "missing_audio_path", "Audio path was not provided.", started, warnings)

            audio_path = Path(request.audio_path)
            if not audio_path.exists():
                return self._error_response("audio", request_id, "audio_file_not_found", "Audio file was not found.", started, warnings, debug_info={"audio_path": str(audio_path)} if request.debug else None)
            audio_report = validate_audio_file(audio_path)
            if not audio_report["supported_extension"]:
                return self._error_response(
                    "audio",
                    request_id,
                    "unsupported_audio_type",
                    "Audio file type is not supported.",
                    started,
                    warnings,
                    debug_info=audio_report if request.debug else None,
                )
            if audio_report["duration_seconds"] is None:
                warnings.append("audio_duration_unreadable")

            audio_quality = analyze_audio_quality(audio_path, self.config.get("audio_quality", {}))
            pause_metrics = dict(audio_quality.get("pause_metrics", {}) or {})
            for warning in audio_quality.get("warnings", []) or []:
                if warning not in warnings:
                    warnings.append(str(warning))
            quality_decision = self._quality_uncertainty_decision(audio_quality, None, expected_text)
            if quality_decision["quality_gate_failed"]:
                analysis = self.run_reading_analysis(expected_text, "", context["accepted_answers"], context["content_metadata"])
                return self._quality_gate_response(
                    request_id=request_id,
                    prompt_id=context.get("prompt_id") or request.prompt_id,
                    expected_text=expected_text,
                    accepted_answers=context["accepted_answers"],
                    analysis=analysis,
                    content_metadata=context["content_metadata"],
                    enrichment_metadata=context["enrichment_metadata"],
                    warnings=warnings,
                    started=started,
                    debug=request.debug,
                    audio_quality=audio_quality,
                    pause_metrics=pause_metrics,
                    uncertainty=quality_decision,
                )

            asr = self.transcribe_audio(str(audio_path), expected_text=expected_text, content_metadata=context["content_metadata"])
            if asr.error:
                warnings.append("transcription_failed")
                return self._error_response("audio", request_id, "transcription_failed", asr.error, started, warnings, debug_info=asr.to_dict() if request.debug else None)

            normalization = normalize_asr_transcript(
                raw_transcript=asr.transcript,
                expected_text=expected_text,
                activity_type=request.activity_type or context["content_metadata"].get("activity_type"),
                prompt_type=request.prompt_type or request.task_type or context["content_metadata"].get("task_type"),
                asr_confidence=asr.confidence,
                cmudict_loader=self.cmudict_loader,
                config=self.config.get("transcript_normalization", {}),
                observed_phonemes=asr.observed_phonemes,
                wav2vec2_transcript=asr.wav2vec2_transcript or asr.transcript,
                model_used=asr.model_used or asr.model_size or "",
                asr_route=asr.asr_route or "wav2vec2_only",
            )
            uncertainty = self._quality_uncertainty_decision(audio_quality, asr, expected_text, normalization)
            gop = compute_gop(
                audio_path_or_waveform=str(audio_path),
                expected_text=expected_text,
                prompt_type=request.prompt_type or request.task_type or context["content_metadata"].get("task_type") or normalization.prompt_type,
                raw_transcript=asr.transcript,
                sample_rate=asr.audio_sample_rate or 16000,
                observed_phonemes=asr.observed_phonemes,
                cmudict_loader=self.cmudict_loader,
                config=self.config.get("gop", {}),
                audio_quality=audio_quality,
                retry_required=bool(uncertainty.get("retry_required", False)),
                uncertain=bool(uncertainty.get("uncertain", False)),
            )
            transcript_meta = apply_gop_to_transcript_meta(normalization.to_dict(), gop)
            transcript_meta = apply_dynamic_expected_word_correction(
                transcript_meta,
                config=self.config.get("dynamic_expected_correction", {}),
                audio_quality=audio_quality,
                retry_required=bool(uncertainty.get("retry_required", False)),
                uncertain=bool(uncertainty.get("uncertain", False)),
                context_metadata=context["content_metadata"],
                cmudict_loader=self.cmudict_loader,
            )
            normalization = TranscriptNormalizationResult(**transcript_meta)
            actual_for_analysis = normalization.corrected_transcript
            if normalization.prompt_type in {"sentence", "paragraph", "passage", "final_sentence", "reading_passage"}:
                actual_for_analysis = normalization.raw_transcript
            analysis = self.run_reading_analysis(expected_text, actual_for_analysis, context["accepted_answers"], context["content_metadata"])
            adaptive = self._maybe_recommend(request.learner_history, request.candidate_items, context, analysis, request.debug)
            return self.build_response(
                request_id=request_id,
                mode="audio",
                prompt_id=context.get("prompt_id") or request.prompt_id,
                expected_text=expected_text,
                accepted_answers=context["accepted_answers"],
                transcript=asr.transcript,
                normalized_transcript=normalize_transcript(normalization.corrected_transcript),
                normalization=normalization,
                confidence=asr.confidence,
                analysis=analysis,
                content_metadata=context["content_metadata"],
                enrichment_metadata=context["enrichment_metadata"],
                warnings=warnings,
                started=started,
                debug=request.debug,
                debug_info={
                    "asr": asr.to_dict(),
                    "audio_quality": audio_quality,
                    "pause_metrics": pause_metrics,
                    "transcript_normalization": normalization.to_dict(),
                    "gop": gop,
                    "learner_response_id": request.learner_response_id,
                    "attempt_id": request.attempt_id,
                },
                adaptive=adaptive,
                audio_quality=audio_quality,
                pause_metrics=pause_metrics,
                uncertainty=uncertainty,
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
            raw_transcript_original=str(result.get("raw_transcript_original", transcript)),
            wav2vec2_transcript=str(result.get("wav2vec2_transcript", transcript)),
            asr_route=str(result.get("asr_route", "wav2vec2_only")),
            model_family=str(result.get("model_family", "wav2vec2")),
            model_used=str(result.get("model_used", self.model_size or "")),
            confidence=result.get("confidence"),
            duration_seconds=result.get("audio_duration_seconds") or result.get("duration_seconds"),
            audio_sample_rate=result.get("audio_sample_rate"),
            provider=str(result.get("provider", self.provider_name)),
            model_size=self.model_size,
            inference_time_ms=result.get("inference_time_ms"),
            observed_phonemes=list(result.get("observed_phonemes", []) or []),
            phoneme_model_used=str(result.get("phoneme_model_used", "")),
            phoneme_inference_time_ms=result.get("phoneme_inference_time_ms"),
            phoneme_error=result.get("phoneme_error"),
            debug_metadata=dict(result.get("debug_metadata", {}) or {}),
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
        normalization: TranscriptNormalizationResult | None = None,
        confidence: float | None = None,
        debug_info: dict[str, Any] | None = None,
        adaptive: dict[str, Any] | None = None,
        audio_quality: dict[str, Any] | None = None,
        pause_metrics: dict[str, Any] | None = None,
        uncertainty: dict[str, Any] | None = None,
        developer_reinforcement: dict[str, Any] | None = None,
    ) -> AnalysisResponse:
        include_debug = debug and bool(self.config.get("api", {}).get("debug", True))
        if include_debug and debug_info is None:
            debug_info = {
                "content_index_path": str(self.content_repository.loaded_path) if self.content_repository.loaded_path else "",
                "asr_provider": self.provider_name,
            }
        merged_enrichment = {**enrichment_metadata}
        transcript_meta = normalization.to_dict() if normalization else {
            "raw_transcript": transcript,
            "corrected_transcript": normalized_transcript or transcript,
            "displayed_transcript": normalized_transcript or transcript,
            "expected_text": expected_text,
            "prompt_type": "unknown",
            "asr_route": "wav2vec2_only",
            "model_family": "wav2vec2",
            "model_used": self.model_size or "",
            "wav2vec2_transcript": transcript,
            "whisper_transcript": None,
            "whisper_removed": True,
            "raw_wer": compute_wer(expected_text, transcript),
            "corrected_wer": compute_wer(expected_text, normalized_transcript or transcript),
            "raw_cer": compute_cer(expected_text, transcript),
            "corrected_cer": compute_cer(expected_text, normalized_transcript or transcript),
            "expected_phonemes": list(analysis.get("expected_phonemes", []) or []),
            "expected_phoneme_source": "",
            "expected_phoneme_variants": [],
            "observed_phonemes": [],
            "phonetic_similarity_score": float(analysis.get("phoneme_similarity", 0.0) or 0.0),
            "composite_score": float(analysis.get("phoneme_similarity", 0.0) or 0.0),
            "accepted": bool(analysis.get("is_correct", False) or analysis.get("is_accepted", False)),
            "normalization_applied": False,
            "normalization_reason": "No ASR transcript correction applied",
            "correction_strategy_used": "none",
            "accepted_by_letter_alias": False,
            "accepted_by_phonetic_threshold": False,
            "accepted_by_known_confusion": False,
            "accepted_by_letter_lattice": False,
            "accepted_by_letter_normalization": False,
            "accepted_by_exact_match": False,
            "accepted_by_vowel_tail": False,
            "accepted_by_phoneme_evidence": False,
            "accepted_by_reinforcement_match": False,
            "reinforcement_source_file": "",
            "reinforcement_expected_label": "",
            "reinforcement_matched_transcript": "",
            "reinforcement_match_normalized": {},
            "reinforcement_match_original": {},
            "critical_phoneme": None,
            "critical_phoneme_detected": None,
            "critical_phoneme_expected_position": None,
            "critical_phoneme_reason": None,
            "critical_pair_detected": False,
            "confidence_level": "",
            "threshold_used": 0.0,
            "confidence_or_threshold_used": 0.0,
            "debug_metadata": {},
        }
        transcript_meta = {**transcript_meta, **gop_response_fields(transcript_meta), **dynamic_response_fields(transcript_meta)}
        if bool(self.config.get("gop", {}).get("debug", False)):
            debug_metadata = dict(transcript_meta.get("debug_metadata", {}) or {})
            debug_metadata["gop"] = {
                key: value
                for key, value in transcript_meta.items()
                if str(key).startswith("gop_") or key in {"mispronounced_phonemes", "weak_words"}
            }
            transcript_meta["debug_metadata"] = debug_metadata
        audio_quality_payload = _json_safe(audio_quality or {})
        pause_metrics_payload = _json_safe(pause_metrics or {})
        uncertainty_payload = uncertainty or {}
        developer_reinforcement_payload = developer_reinforcement or {}
        retry_required = bool(uncertainty_payload.get("retry_required", False))
        raw_transcript = str(transcript_meta.get("raw_transcript", transcript))
        corrected_transcript = str(transcript_meta.get("corrected_transcript", normalized_transcript or transcript))
        displayed_transcript = str(transcript_meta.get("displayed_transcript", transcript_meta.get("corrected_transcript", normalized_transcript or transcript)))
        if retry_required:
            displayed_transcript = raw_transcript
            corrected_transcript = raw_transcript
        response = AnalysisResponse(
            ok=True,
            request_id=request_id,
            mode=mode,
            provider=self.provider_name,
            model_size=self.model_size,
            prompt_id=prompt_id,
            expected_text=expected_text,
            accepted_answers=accepted_answers,
            transcript=displayed_transcript,
            normalized_transcript=normalized_transcript,
            raw_transcript=raw_transcript,
            corrected_transcript=corrected_transcript,
            displayed_transcript=displayed_transcript,
            prompt_type=str(transcript_meta.get("prompt_type", "unknown")),
            asr_route=str(transcript_meta.get("asr_route", "wav2vec2_only")),
            model_family=str(transcript_meta.get("model_family", "wav2vec2")),
            model_used=str(transcript_meta.get("model_used", self.model_size or "")),
            wav2vec2_transcript=str(transcript_meta.get("wav2vec2_transcript", transcript)),
            whisper_transcript=None,
            whisper_removed=bool(transcript_meta.get("whisper_removed", True)),
            raw_wer=float(transcript_meta.get("raw_wer", 0.0) or 0.0),
            corrected_wer=float(transcript_meta.get("corrected_wer", 0.0) or 0.0),
            raw_cer=float(transcript_meta.get("raw_cer", 0.0) or 0.0),
            corrected_cer=float(transcript_meta.get("corrected_cer", 0.0) or 0.0),
            phonetic_similarity_score=float(transcript_meta.get("phonetic_similarity_score", 0.0) or 0.0),
            composite_score=float(transcript_meta.get("composite_score", 0.0) or 0.0),
            accepted=False if retry_required else bool(transcript_meta.get("accepted", False)),
            normalization_applied=bool(transcript_meta.get("normalization_applied", False)),
            normalization_reason=str(transcript_meta.get("normalization_reason", "")),
            correction_strategy_used=str(transcript_meta.get("correction_strategy_used", "none")),
            accepted_by_letter_alias=bool(transcript_meta.get("accepted_by_letter_alias", False)),
            accepted_by_phonetic_threshold=bool(transcript_meta.get("accepted_by_phonetic_threshold", False)),
            accepted_by_known_confusion=bool(transcript_meta.get("accepted_by_known_confusion", False)),
            accepted_by_letter_lattice=bool(transcript_meta.get("accepted_by_letter_lattice", False)),
            accepted_by_letter_normalization=bool(transcript_meta.get("accepted_by_letter_normalization", False)),
            accepted_by_exact_match=bool(transcript_meta.get("accepted_by_exact_match", False)),
            accepted_by_vowel_tail=bool(transcript_meta.get("accepted_by_vowel_tail", False)),
            accepted_by_phoneme_evidence=bool(transcript_meta.get("accepted_by_phoneme_evidence", False)),
            gop_enabled=bool(transcript_meta.get("gop_enabled", True)),
            gop_available=bool(transcript_meta.get("gop_available", False)),
            gop_score=transcript_meta.get("gop_score"),
            gop_confidence=transcript_meta.get("gop_confidence"),
            gop_decision=str(transcript_meta.get("gop_decision", "not_available")),
            gop_threshold=transcript_meta.get("gop_threshold"),
            gop_prompt_type=str(transcript_meta.get("gop_prompt_type", "unknown")),
            gop_expected_phonemes=list(transcript_meta.get("gop_expected_phonemes", []) or []),
            gop_observed_phonemes=list(transcript_meta.get("gop_observed_phonemes", []) or []),
            gop_phoneme_scores=list(transcript_meta.get("gop_phoneme_scores", []) or []),
            gop_word_scores=list(transcript_meta.get("gop_word_scores", []) or []),
            mispronounced_phonemes=list(transcript_meta.get("mispronounced_phonemes", []) or []),
            weak_words=list(transcript_meta.get("weak_words", []) or []),
            gop_correction_applied=False if retry_required else bool(transcript_meta.get("gop_correction_applied", False)),
            gop_error=transcript_meta.get("gop_error"),
            dynamic_correction_enabled=bool(transcript_meta.get("dynamic_correction_enabled", True)),
            dynamic_correction_applied=False if retry_required else bool(transcript_meta.get("dynamic_correction_applied", False)),
            dynamic_correction_strategy=str(transcript_meta.get("dynamic_correction_strategy", "dynamic_expected_word_correction")),
            dynamic_correction_sub_strategy=str(transcript_meta.get("dynamic_correction_sub_strategy", "")),
            dynamic_correction_confidence=transcript_meta.get("dynamic_correction_confidence"),
            dynamic_correction_threshold=transcript_meta.get("dynamic_correction_threshold"),
            dynamic_spelling_similarity=transcript_meta.get("dynamic_spelling_similarity"),
            dynamic_phoneme_similarity=transcript_meta.get("dynamic_phoneme_similarity"),
            dynamic_gop_score=transcript_meta.get("dynamic_gop_score"),
            dynamic_homophone_match=bool(transcript_meta.get("dynamic_homophone_match", False)),
            dynamic_context_score=transcript_meta.get("dynamic_context_score"),
            dynamic_correction_reason=str(transcript_meta.get("dynamic_correction_reason", "")),
            dynamic_suspicious_fragment=bool(transcript_meta.get("dynamic_suspicious_fragment", False)),
            dynamic_fragment_reasons=list(transcript_meta.get("dynamic_fragment_reasons", []) or []),
            dynamic_phoneme_coverage=transcript_meta.get("dynamic_phoneme_coverage"),
            asr_spelling_variant_enabled=bool(transcript_meta.get("asr_spelling_variant_enabled", True)),
            asr_spelling_variant_applied=False if retry_required else bool(transcript_meta.get("asr_spelling_variant_applied", False)),
            asr_spelling_variant_strategy=str(transcript_meta.get("asr_spelling_variant_strategy", "dynamic_asr_spelling_variant")),
            asr_spelling_variant_sub_strategy=str(transcript_meta.get("asr_spelling_variant_sub_strategy", "")),
            asr_spelling_variant_confidence=transcript_meta.get("asr_spelling_variant_confidence"),
            asr_spelling_variant_threshold=transcript_meta.get("asr_spelling_variant_threshold"),
            consonant_skeleton_similarity=transcript_meta.get("consonant_skeleton_similarity"),
            vowel_tolerant_similarity=transcript_meta.get("vowel_tolerant_similarity"),
            expected_phoneme_coverage=transcript_meta.get("expected_phoneme_coverage"),
            variant_edit_similarity=transcript_meta.get("variant_edit_similarity"),
            variant_reason=str(transcript_meta.get("variant_reason", "")),
            word_alignment=list(transcript_meta.get("word_alignment", []) or []),
            accepted_by_reinforcement_match=bool(transcript_meta.get("accepted_by_reinforcement_match", False)),
            reinforcement_source_file=str(transcript_meta.get("reinforcement_source_file", "")),
            reinforcement_expected_label=str(transcript_meta.get("reinforcement_expected_label", "")),
            reinforcement_matched_transcript=str(transcript_meta.get("reinforcement_matched_transcript", "")),
            reinforcement_match_normalized=dict(transcript_meta.get("reinforcement_match_normalized", {}) or {}),
            reinforcement_match_original=dict(transcript_meta.get("reinforcement_match_original", {}) or {}),
            critical_phoneme=transcript_meta.get("critical_phoneme"),
            critical_phoneme_detected=transcript_meta.get("critical_phoneme_detected"),
            critical_phoneme_expected_position=transcript_meta.get("critical_phoneme_expected_position"),
            critical_phoneme_reason=transcript_meta.get("critical_phoneme_reason"),
            critical_pair_detected=bool(transcript_meta.get("critical_pair_detected", False)),
            confidence_level=str(transcript_meta.get("confidence_level", "")),
            threshold_used=float(transcript_meta.get("threshold_used", 0.0) or 0.0),
            confidence_or_threshold_used=float(transcript_meta.get("confidence_or_threshold_used", 0.0) or 0.0),
            confidence=confidence,
            is_correct=False if retry_required else bool(analysis.get("is_correct", False)),
            is_exact=False if retry_required else bool(analysis.get("is_exact", False)),
            is_accepted=False if retry_required else bool(analysis.get("is_accepted", False)),
            character_similarity=float(analysis.get("character_similarity", 0.0) or 0.0),
            token_similarity=float(analysis.get("token_similarity", 0.0) or 0.0),
            similarity_label=str(analysis.get("similarity_label", "")),
            expected_phonemes=list(transcript_meta.get("expected_phonemes", analysis.get("expected_phonemes", [])) or []),
            expected_phoneme_source=str(transcript_meta.get("expected_phoneme_source", "")),
            expected_phoneme_variants=list(transcript_meta.get("expected_phoneme_variants", []) or []),
            observed_phonemes=list(transcript_meta.get("observed_phonemes", []) or []),
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
            audio_quality=audio_quality_payload,
            pause_metrics=pause_metrics_payload,
            uncertain=bool(uncertainty_payload.get("uncertain", False)),
            retry_required=retry_required,
            uncertainty_reasons=list(uncertainty_payload.get("uncertainty_reasons", []) or []),
            quality_gate_failed=bool(uncertainty_payload.get("quality_gate_failed", False)),
            learner_retry_message=str(uncertainty_payload.get("learner_retry_message", "")),
            developer_quality_notes=list(uncertainty_payload.get("developer_quality_notes", []) or []),
            content_metadata=_json_safe(content_metadata),
            enrichment_metadata=_json_safe(merged_enrichment),
            analysis_source=str(analysis.get("analysis_source", "heuristic_transcript_phoneme")),
            debug_metadata=dict(transcript_meta.get("debug_metadata", {}) or {}),
            developer_reinforcement_mode=bool(developer_reinforcement_payload.get("mode", False)),
            reinforcement_saved=bool(developer_reinforcement_payload.get("saved", False)),
            reinforcement_duplicate=bool(developer_reinforcement_payload.get("duplicate", False)),
            reinforcement_target_file=str(developer_reinforcement_payload.get("target_file", "")),
            reinforcement_reason=str(developer_reinforcement_payload.get("reason", "")),
            warnings=warnings,
            debug_info=_json_safe(debug_info) if include_debug else None,
            processing_seconds=round(time.perf_counter() - started, 3),
            error=None,
        )
        logger.info("request_id=%s endpoint=%s provider=%s prompt_id=%s ok=true", request_id, mode, self.provider_name, prompt_id)
        return response

    def _quality_uncertainty_decision(
        self,
        audio_quality: dict[str, Any] | None,
        asr: ASRResult | None,
        expected_text: str,
        normalization: TranscriptNormalizationResult | None = None,
    ) -> dict[str, Any]:
        qa_config = audio_quality_config(self.config.get("audio_quality", {}))
        reasons: list[str] = []
        notes: list[str] = []
        quality = audio_quality or {}
        flags = dict(quality.get("quality_flags", {}) or {})
        reason_map = {
            "too_short": "audio_too_short",
            "no_speech_detected": "no_speech_detected",
            "mostly_silent": "mostly_silent",
            "low_volume": "low_volume",
            "clipped": "clipped",
        }
        for flag, reason in reason_map.items():
            if flags.get(flag):
                reasons.append(reason)

        if asr is not None:
            transcript = str(asr.transcript or "").strip()
            observed_phonemes = list(asr.observed_phonemes or [])
            if not transcript and not observed_phonemes:
                reasons.append("blank_asr_transcript")
                if expected_text:
                    reasons.append("expected_text_without_reliable_asr_evidence")
            if asr.confidence is not None and asr.confidence < float(self.config.get("transcript_normalization", {}).get("low_confidence_threshold", 0.50)):
                reasons.append("low_asr_confidence")
            if normalization and str(normalization.confidence_level).lower().startswith("low"):
                reasons.append("low_normalization_confidence")

        unique_reasons = list(dict.fromkeys(reasons))
        retry_required = bool(unique_reasons and qa_config["retry_on_bad_quality"])
        quality_gate_failed = bool(
            qa_config["enable_quality_gate"]
            and qa_config["retry_on_bad_quality"]
            and any(flags.get(flag) for flag in reason_map)
        )
        if quality.get("warnings"):
            notes.extend(str(warning) for warning in quality.get("warnings", []) or [])
        if quality_gate_failed:
            notes.append("ASR skipped because strict audio quality gate is enabled.")
        elif retry_required:
            notes.append("ASR result is marked retry-required because quality or confidence was unreliable.")

        return {
            "uncertain": bool(unique_reasons),
            "retry_required": retry_required,
            "uncertainty_reasons": unique_reasons,
            "quality_gate_failed": quality_gate_failed,
            "learner_retry_message": _learner_retry_message(unique_reasons),
            "developer_quality_notes": notes,
        }

    def _quality_gate_response(
        self,
        request_id: str,
        prompt_id: str | None,
        expected_text: str,
        accepted_answers: list[str],
        analysis: dict[str, Any],
        content_metadata: dict[str, Any],
        enrichment_metadata: dict[str, Any],
        warnings: list[str],
        started: float,
        debug: bool,
        audio_quality: dict[str, Any],
        pause_metrics: dict[str, Any],
        uncertainty: dict[str, Any],
    ) -> AnalysisResponse:
        include_debug = debug and bool(self.config.get("api", {}).get("debug", True))
        debug_info = {"audio_quality": audio_quality, "pause_metrics": pause_metrics} if include_debug else None
        reason = _normalization_reason_for_quality(uncertainty.get("uncertainty_reasons", []))
        gop = gop_response_fields({
            "gop_enabled": bool(self.config.get("gop", {}).get("enabled", True)),
            "gop_available": False,
            "gop_decision": "skipped_bad_audio",
            "gop_error": reason,
        })
        logger.info("request_id=%s endpoint=audio provider=%s prompt_id=%s quality_gate_failed=true", request_id, self.provider_name, prompt_id)
        return AnalysisResponse(
            ok=True,
            request_id=request_id,
            mode="audio",
            provider=self.provider_name,
            model_size=self.model_size,
            prompt_id=prompt_id,
            expected_text=expected_text,
            accepted_answers=accepted_answers,
            transcript="",
            normalized_transcript="",
            raw_transcript="",
            corrected_transcript="",
            displayed_transcript="",
            prompt_type="unknown",
            asr_route="wav2vec2_only",
            model_family="wav2vec2",
            model_used=self.model_size or "",
            wav2vec2_transcript="",
            whisper_transcript=None,
            whisper_removed=True,
            accepted=False,
            normalization_applied=False,
            normalization_reason=reason,
            correction_strategy_used="audio_quality_gate",
            **gop,
            is_correct=False,
            is_exact=False,
            is_accepted=False,
            character_similarity=float(analysis.get("character_similarity", 0.0) or 0.0),
            token_similarity=float(analysis.get("token_similarity", 0.0) or 0.0),
            similarity_label=str(analysis.get("similarity_label", "blank")),
            expected_phonemes=list(analysis.get("expected_phonemes", []) or []),
            actual_phonemes=list(analysis.get("actual_phonemes", []) or []),
            phoneme_similarity=float(analysis.get("phoneme_similarity", 0.0) or 0.0),
            error_type=str(analysis.get("error_type", "")),
            feedback_hint=str(analysis.get("feedback_hint", "")),
            coach_hint_key=str(analysis.get("coach_hint_key", "")),
            learner_safe_summary=str(analysis.get("learner_safe_summary", "")),
            skill_signal=str(analysis.get("skill_signal", "")),
            target_phoneme=str(analysis.get("target_phoneme", "")),
            target_position=str(analysis.get("target_position", "")),
            recommended_practice_focus=str(analysis.get("recommended_practice_focus", "")),
            recommended_action=str(analysis.get("recommended_action", "")),
            audio_quality=_json_safe(audio_quality),
            pause_metrics=_json_safe(pause_metrics),
            uncertain=True,
            retry_required=True,
            uncertainty_reasons=list(uncertainty.get("uncertainty_reasons", []) or []),
            quality_gate_failed=True,
            learner_retry_message=str(uncertainty.get("learner_retry_message", "")),
            developer_quality_notes=list(uncertainty.get("developer_quality_notes", []) or []),
            content_metadata=_json_safe(content_metadata),
            enrichment_metadata=_json_safe(enrichment_metadata),
            analysis_source=str(analysis.get("analysis_source", "heuristic_transcript_phoneme")),
            debug_metadata={},
            warnings=warnings,
            debug_info=_json_safe(debug_info) if include_debug else None,
            processing_seconds=round(time.perf_counter() - started, 3),
            error=None,
        )

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


def _learner_retry_message(reasons: list[str]) -> str:
    if not reasons:
        return ""
    if "audio_too_short" in reasons:
        return "Please record again. Make sure your voice is clear and at least 1 second long."
    if "no_speech_detected" in reasons or "mostly_silent" in reasons:
        return "Please record again. Make sure your voice is clear and close enough to the microphone."
    if "low_volume" in reasons:
        return "Please record again with a clearer, louder voice."
    if "clipped" in reasons:
        return "Please record again a little farther from the microphone."
    return "Please record again. The audio was not reliable enough to score."


def _normalization_reason_for_quality(reasons: list[str]) -> str:
    if "audio_too_short" in reasons:
        return "Audio is too short for reliable ASR."
    if "no_speech_detected" in reasons:
        return "No speech was detected in the recording."
    if "mostly_silent" in reasons:
        return "Audio is mostly silent and cannot be scored reliably."
    if "low_volume" in reasons:
        return "Audio volume is too low for reliable ASR."
    if "clipped" in reasons:
        return "Audio appears clipped or distorted."
    return "Audio quality was not reliable enough for scoring."


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
