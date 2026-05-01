from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from readirect_asr.evaluation.model_comparison import (
    aggregate_rows,
    correct_expected_centric,
    infer_prompt_type,
    normalize_eval_text,
    recommendation_from_summary,
    score_pair,
    winner_for,
    write_csv,
    write_jsonl,
    write_markdown_report,
    write_summary_json,
)
from readirect_asr.finetuning.whisper_generation_config import prepare_whisper_generation_config
from training.wav2vec2_manifest_utils import read_jsonl, resolve_repo_path


DEFAULT_WAV2VEC2_MODEL = "models/wav2vec2-readirect-asr"
DEFAULT_BASE_WAV2VEC2_MODEL = "models/wav2vec2-base-960h"
DEFAULT_WHISPER_MODEL = "model_artifacts/readirect-whisper-base-en-v1-hf"
OUTPUT_FILES = (
    "wav2vec2_vs_whisper_rows.csv",
    "wav2vec2_vs_whisper_rows.jsonl",
    "wav2vec2_vs_whisper_summary.json",
    "wav2vec2_vs_whisper_report.md",
    "wav2vec2_vs_whisper_error_analysis.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair side-by-side Wav2Vec2 vs Whisper evaluation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--wav2vec2-model", default=DEFAULT_WAV2VEC2_MODEL, type=Path)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/evaluation/model_comparison"), type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-type", choices=("letter", "word", "sentence", "all"), default="all")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--use-correction", dest="use_correction", action="store_true", default=True)
    parser.add_argument("--no-correction", dest="use_correction", action="store_false")
    parser.add_argument("--use-phoneme-evidence", dest="use_phoneme_evidence", action="store_true", default=False)
    parser.add_argument("--no-phoneme-evidence", dest="use_phoneme_evidence", action="store_false")
    parser.add_argument("--allow-base-wav2vec2-fallback", action="store_true")
    parser.add_argument("--allow-whisper-api-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class Wav2Vec2Evaluator:
    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(str(model_path), local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = sample_rate

    def transcribe(self, audio_path: str) -> str:
        import librosa

        audio, sr = librosa.load(str(resolve_repo_path(audio_path)), sr=self.sample_rate, mono=True)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        predicted_ids = self.torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0].strip()


class WhisperEvaluator:
    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(str(model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        prepare_whisper_generation_config(self.model, self.processor, language="en", task="transcribe", verbose=False)
        self.sample_rate = sample_rate

    def transcribe(self, audio_path: str) -> str:
        import librosa

        audio, sr = librosa.load(str(resolve_repo_path(audio_path)), sr=self.sample_rate, mono=True)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)
        with self.torch.no_grad():
            predicted_ids = self.model.generate(inputs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def resolve_wav2vec2_path(args: argparse.Namespace) -> Path:
    requested = resolve_repo_path(args.wav2vec2_model)
    if requested.exists():
        return requested
    fallback = resolve_repo_path(DEFAULT_BASE_WAV2VEC2_MODEL)
    if args.allow_base_wav2vec2_fallback and fallback.exists():
        print(f"Warning: fine-tuned Wav2Vec2 missing; using fallback {fallback.relative_to(PROJECT_ROOT)}")
        return fallback
    raise FileNotFoundError(f"Wav2Vec2 model not found: {requested}")


def resolve_whisper_path(args: argparse.Namespace) -> Path:
    requested = resolve_repo_path(args.whisper_model)
    if requested.exists():
        return requested
    if args.allow_whisper_api_fallback:
        api_default = resolve_repo_path(DEFAULT_WHISPER_MODEL)
        if api_default.exists():
            print(f"Warning: requested Whisper model missing; using API default {api_default.relative_to(PROJECT_ROOT)}")
            return api_default
    raise FileNotFoundError(
        f"Whisper model not found: {requested}. Pass --whisper-model or install the fine-tuned model artifact."
    )


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        expected = str(row.get("text", "")).strip()
        prompt_type = str(row.get("prompt_type") or row.get("metadata", {}).get("prompt_type") or infer_prompt_type(expected))
        dataset = str(row.get("dataset", "") or "unknown")
        if args.prompt_type != "all" and prompt_type != args.prompt_type:
            continue
        if args.dataset != "all" and dataset != args.dataset:
            continue
        row = dict(row)
        row["_inferred_prompt_type"] = prompt_type
        selected.append(row)
        if args.max_samples is not None and len(selected) >= args.max_samples:
            break
    return selected


def output_dir_for_run(output_dir: Path, overwrite: bool) -> Path:
    resolved = resolve_repo_path(output_dir)
    if overwrite or not any((resolved / name).exists() for name in OUTPUT_FILES):
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    timestamped = resolved / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    timestamped.mkdir(parents=True, exist_ok=False)
    print(f"Existing comparison outputs found. Writing this run to {timestamped.relative_to(PROJECT_ROOT)}")
    return timestamped


def model_result(
    prefix: str,
    raw_transcript: str,
    expected: str,
    prompt_type: str,
    use_correction: bool,
    use_phoneme_evidence: bool,
) -> dict[str, Any]:
    normalized_raw = normalize_eval_text(raw_transcript)
    raw_score = score_pair(expected, normalized_raw)
    correction = correct_expected_centric(
        expected,
        raw_transcript,
        prompt_type,
        use_correction=use_correction,
        use_phoneme_evidence=use_phoneme_evidence,
    )
    return {
        f"{prefix}_raw_transcript": raw_transcript,
        f"{prefix}_normalized_transcript": normalized_raw,
        f"{prefix}_corrected_transcript": correction.corrected_transcript,
        f"{prefix}_displayed_transcript": correction.displayed_transcript,
        f"{prefix}_raw_wer": raw_score["wer"],
        f"{prefix}_corrected_wer": correction.corrected_wer,
        f"{prefix}_raw_cer": raw_score["cer"],
        f"{prefix}_corrected_cer": correction.corrected_cer,
        f"{prefix}_exact_match": raw_score["exact_match"],
        f"{prefix}_corrected_exact_match": correction.corrected_exact_match,
        f"{prefix}_accepted": correction.accepted,
        f"{prefix}_correction_reason": correction.reason,
        f"{prefix}_error": "",
    }


def error_model_result(prefix: str, error: str) -> dict[str, Any]:
    return {
        f"{prefix}_raw_transcript": "",
        f"{prefix}_normalized_transcript": "",
        f"{prefix}_corrected_transcript": "",
        f"{prefix}_displayed_transcript": "",
        f"{prefix}_raw_wer": 1.0,
        f"{prefix}_corrected_wer": 1.0,
        f"{prefix}_raw_cer": 1.0,
        f"{prefix}_corrected_cer": 1.0,
        f"{prefix}_exact_match": False,
        f"{prefix}_corrected_exact_match": False,
        f"{prefix}_accepted": False,
        f"{prefix}_correction_reason": "model_error",
        f"{prefix}_error": error,
    }


def compare_row(
    index: int,
    row: dict[str, Any],
    wav2vec2: Wav2Vec2Evaluator,
    whisper: WhisperEvaluator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    audio_path = str(row.get("audio_path", "")).strip()
    expected = normalize_eval_text(row.get("text", ""))
    prompt_type = str(row.get("_inferred_prompt_type") or infer_prompt_type(expected))
    output = {
        "row_id": index,
        "audio_path": audio_path,
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "speaker_id": row.get("speaker_id", ""),
        "source_id": row.get("source_id", ""),
        "prompt_type": prompt_type,
        "expected_text": row.get("text", ""),
        "normalized_expected": expected,
    }
    if not audio_path or not resolve_repo_path(audio_path).exists():
        output.update(error_model_result("wav2vec2", "audio_file_not_found"))
        output.update(error_model_result("whisper", "audio_file_not_found"))
    else:
        try:
            print(f"[{index}] wav2vec2 {row.get('dataset', '')}/{prompt_type}: {row.get('source_id', '')}")
            output.update(model_result("wav2vec2", wav2vec2.transcribe(audio_path), expected, prompt_type, args.use_correction, args.use_phoneme_evidence))
        except Exception as exc:
            output.update(error_model_result("wav2vec2", str(exc)))
        try:
            print(f"[{index}] whisper   {row.get('dataset', '')}/{prompt_type}: {row.get('source_id', '')}")
            output.update(model_result("whisper", whisper.transcribe(audio_path), expected, prompt_type, args.use_correction, args.use_phoneme_evidence))
        except Exception as exc:
            output.update(error_model_result("whisper", str(exc)))

    output["winner_raw"] = winner_for(float(output["wav2vec2_raw_wer"]), float(output["whisper_raw_wer"]))
    output["winner_corrected"] = winner_for(float(output["wav2vec2_corrected_wer"]), float(output["whisper_corrected_wer"]))
    return output


def build_error_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("wav2vec2_error") or row.get("whisper_error") or row.get("winner_corrected") != "tie":
            error_rows.append(
                {
                    "row_id": row["row_id"],
                    "dataset": row.get("dataset"),
                    "prompt_type": row.get("prompt_type"),
                    "source_id": row.get("source_id"),
                    "expected": row.get("normalized_expected"),
                    "wav2vec2_prediction": row.get("wav2vec2_normalized_transcript"),
                    "whisper_prediction": row.get("whisper_normalized_transcript"),
                    "wav2vec2_corrected_wer": row.get("wav2vec2_corrected_wer"),
                    "whisper_corrected_wer": row.get("whisper_corrected_wer"),
                    "winner_corrected": row.get("winner_corrected"),
                    "wav2vec2_error": row.get("wav2vec2_error"),
                    "whisper_error": row.get("whisper_error"),
                }
            )
    return error_rows


def build_summary(args: argparse.Namespace, rows: list[dict[str, Any]], output_dir: Path, wav_path: Path, whisper_path: Path) -> dict[str, Any]:
    summary = {
        "manifest": str(resolve_repo_path(args.manifest)),
        "wav2vec2_model": str(wav_path),
        "whisper_model": str(whisper_path),
        "output_dir": str(output_dir),
        "rows_requested": len(rows),
        "rows_evaluated": len(rows),
        "fair_comparison": True,
        "use_correction": bool(args.use_correction),
        "phoneme_evidence": "not_used_transcript_only" if not args.use_phoneme_evidence else "requested_not_implemented",
        "overall": aggregate_rows(rows),
        "by_prompt_type": aggregate_rows(rows, "prompt_type"),
        "by_dataset": aggregate_rows(rows, "dataset"),
        "skipped_or_error_rows": sum(1 for row in rows if row.get("wav2vec2_error") or row.get("whisper_error")),
        "notes": [
            "Both models were evaluated on the same manifest rows selected by the CLI filters.",
            "Both models use the same evaluation normalizer and WER/CER scoring functions.",
            "Correction is expected-centric and transcript-only unless phoneme evidence is implemented later.",
            "Q vs you is rejected in transcript-only correction because the initial K phoneme cannot be verified from transcript alone.",
        ],
    }
    summary["recommendation"] = recommendation_from_summary(summary)
    return summary


def main() -> int:
    args = parse_args()
    wav_path = resolve_wav2vec2_path(args)
    whisper_path = resolve_whisper_path(args)
    output_dir = output_dir_for_run(args.output_dir, args.overwrite)
    manifest_rows = read_jsonl(args.manifest)
    rows = select_rows(manifest_rows, args)
    if not rows:
        raise RuntimeError("No rows selected for comparison. Check manifest, --prompt-type, and --dataset filters.")
    print(f"Selected {len(rows)} rows from {args.manifest}")
    print(f"Wav2Vec2 model: {wav_path}")
    print(f"Whisper model: {whisper_path}")

    started = time.perf_counter()
    wav2vec2 = Wav2Vec2Evaluator(wav_path)
    whisper = WhisperEvaluator(whisper_path)

    comparison_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        comparison_rows.append(compare_row(index, row, wav2vec2, whisper, args))
        if index % 10 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(f"Progress: {index}/{len(rows)} rows, elapsed={elapsed:.1f}s")

    error_analysis = build_error_analysis(comparison_rows)
    summary = build_summary(args, comparison_rows, output_dir, wav_path, whisper_path)

    write_csv(output_dir / "wav2vec2_vs_whisper_rows.csv", comparison_rows)
    write_jsonl(output_dir / "wav2vec2_vs_whisper_rows.jsonl", comparison_rows)
    write_summary_json(output_dir / "wav2vec2_vs_whisper_summary.json", summary)
    write_markdown_report(output_dir / "wav2vec2_vs_whisper_report.md", summary)
    write_csv(output_dir / "wav2vec2_vs_whisper_error_analysis.csv", error_analysis)

    print(f"Completed comparison in {time.perf_counter() - started:.1f}s")
    print(f"Outputs saved to {output_dir.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
