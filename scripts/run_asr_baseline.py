from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.asr.faster_whisper_asr import FasterWhisperASR
from readirect_asr.asr.mock_asr import MockASR
from readirect_asr.asr.result import ASRResult
from readirect_asr.text.normalization import normalize_transcript


OUTPUT_COLUMNS = [
    "asr_transcript",
    "normalized_transcript",
    "asr_provider",
    "asr_model_size",
    "asr_processing_seconds",
    "asr_error",
]


def resolve_audio_path(audio_path: str, audio_base: Path | None = None) -> Path:
    path = Path(str(audio_path))
    if path.is_absolute() or path.exists():
        return path
    if audio_base:
        return audio_base / path
    return path


def create_provider(
    provider: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
    beam_size: int,
) -> Any:
    if provider == "mock":
        return MockASR()
    if provider in {"faster_whisper", "faster-whisper"}:
        return FasterWhisperASR(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
        )
    raise ValueError(f"Unsupported ASR provider: {provider}")


def normalize_result(result: ASRResult | dict[str, Any], provider: str, model_size: str) -> dict[str, object]:
    if isinstance(result, ASRResult):
        return {
            "asr_transcript": result.transcript,
            "normalized_transcript": result.normalized_transcript,
            "asr_provider": result.provider,
            "asr_model_size": result.model_size,
            "asr_processing_seconds": result.processing_seconds,
            "asr_error": result.error or "",
        }
    transcript = str(result.get("transcript", ""))
    return {
        "asr_transcript": transcript,
        "normalized_transcript": normalize_transcript(transcript),
        "asr_provider": str(result.get("provider", provider)),
        "asr_model_size": model_size,
        "asr_processing_seconds": result.get("processing_seconds", ""),
        "asr_error": result.get("error", ""),
    }


def run_baseline(
    manifest: Path,
    output: Path,
    provider_name: str = "faster_whisper",
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 1,
    limit: int | None = None,
    start_index: int = 0,
    resume: bool = False,
    audio_base: Path | None = None,
    save_every: int = 25,
) -> pd.DataFrame:
    if resume and output.exists():
        df = pd.read_csv(output)
        attempted_indices = list(range(start_index, len(df)))
        if limit is not None:
            attempted_indices = attempted_indices[:limit]
    else:
        source_df = pd.read_csv(manifest)
        end_index = start_index + limit if limit is not None else None
        df = source_df.iloc[start_index:end_index].copy().reset_index(drop=True)
        attempted_indices = list(range(len(df)))

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object").where(pd.notna(df[column]), "")

    provider = create_provider(provider_name, model_size, device, compute_type, language, beam_size)
    if provider_name in {"faster_whisper", "faster-whisper"} and hasattr(provider, "is_available") and not provider.is_available():
        print("faster-whisper is not installed. Install it with: pip install faster-whisper")
        print("Rows will receive ASR errors until the dependency is installed.")
    processed = 0

    for count, index in enumerate(attempted_indices, start=1):
        if resume and str(df.at[index, "asr_transcript"]).strip():
            continue
        audio_path = resolve_audio_path(str(df.at[index, "audio_path"]), audio_base)
        if not audio_path.exists():
            df.at[index, "asr_error"] = f"audio file not found: {audio_path}"
            df.at[index, "asr_provider"] = provider_name
            df.at[index, "asr_model_size"] = model_size
        else:
            try:
                if provider_name == "mock":
                    expected = str(df.at[index, "manual_transcript"] or df.at[index, "expected_text"])
                    result = provider.transcribe(str(audio_path), expected_text=expected)
                else:
                    result = provider.transcribe(str(audio_path), language=language, beam_size=beam_size)
                for key, value in normalize_result(result, provider_name, model_size).items():
                    df.at[index, key] = value
            except Exception as exc:
                df.at[index, "asr_error"] = str(exc)
                df.at[index, "asr_provider"] = provider_name
                df.at[index, "asr_model_size"] = model_size
        processed += 1
        if save_every > 0 and count % save_every == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Rows attempted: {processed}")
    print(f"Rows with ASR transcript: {int(df['asr_transcript'].fillna('').astype(str).str.strip().ne('').sum())}")
    print(f"Rows with ASR errors: {int(df['asr_error'].fillna('').astype(str).str.strip().ne('').sum())}")
    print(f"Output path: {output}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline ASR over a manifest.")
    parser.add_argument("--manifest", default="data/manifests/speechocean762_manifest.csv", type=Path)
    parser.add_argument("--output", default="data/manifests/speechocean762_asr_baseline.csv", type=Path)
    parser.add_argument("--model-size", default="base.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="en")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", default="faster_whisper")
    parser.add_argument("--audio-base", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--beam-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        manifest=args.manifest,
        output=args.output,
        provider_name=args.provider,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        limit=args.limit,
        start_index=args.start_index,
        resume=args.resume,
        audio_base=args.audio_base,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
