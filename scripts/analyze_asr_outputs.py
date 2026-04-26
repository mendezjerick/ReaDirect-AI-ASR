from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readirect_asr.phonemes.cmudict_loader import CMUDictLoader
from readirect_asr.scoring.reading_analyzer import analyze_reading_response


def choose_reference(row: pd.Series, reference_col: str) -> str:
    if reference_col != "auto":
        return str(row.get(reference_col, "") if pd.notna(row.get(reference_col, "")) else "")
    manual = str(row.get("manual_transcript", "") if pd.notna(row.get("manual_transcript", "")) else "").strip()
    return manual or str(row.get("expected_text", "") if pd.notna(row.get("expected_text", "")) else "")


def analyze_file(
    input_path: Path,
    output_path: Path,
    cmudict_dir: Path,
    reference_col: str = "auto",
    hypothesis_col: str = "normalized_transcript",
    limit: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit).copy()
    loader = CMUDictLoader(
        cmudict_dir / "cmudict.dict",
        cmudict_dir / "cmudict.phones",
        cmudict_dir / "cmudict.symbols",
    ).load()
    rows = []
    for _, row in df.iterrows():
        expected = choose_reference(row, reference_col)
        actual = str(row.get(hypothesis_col, "") if pd.notna(row.get(hypothesis_col, "")) else "")
        analysis = analyze_reading_response(
            expected_text=expected,
            actual_text=actual,
            accepted_answers=row.get("accepted_answers", ""),
            cmudict_loader=loader,
            content_metadata=row.to_dict(),
        )
        rows.append(
            {
                "analysis_is_correct": analysis["is_correct"],
                "analysis_similarity_label": analysis["similarity_label"],
                "analysis_character_similarity": analysis["character_similarity"],
                "analysis_token_similarity": analysis["token_similarity"],
                "analysis_expected_phonemes": " ".join(analysis["expected_phonemes"]),  # type: ignore[arg-type]
                "analysis_actual_phonemes": " ".join(analysis["actual_phonemes"]),  # type: ignore[arg-type]
                "analysis_phoneme_similarity": analysis["phoneme_similarity"],
                "analysis_error_type": analysis["error_type"],
                "analysis_error_position": analysis["error_position"],
                "analysis_feedback_hint": analysis["feedback_hint"],
                "analysis_coach_hint_key": analysis["coach_hint_key"],
                "analysis_skill_signal": analysis["skill_signal"],
                "analysis_target_phoneme": analysis["target_phoneme"],
                "analysis_recommended_practice_focus": analysis["recommended_practice_focus"],
                "analysis_recommended_action": analysis["recommended_action"],
            }
        )
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")
    for column in ("analysis_error_type", "analysis_skill_signal", "analysis_similarity_label"):
        if column in out.columns:
            print(f"\n{column}:")
            print(out[column].fillna("").astype(str).value_counts().to_string())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ASR outputs with ReaDirect reading heuristics.")
    parser.add_argument("--input", default="data/manifests/speechocean762_asr_baseline.csv", type=Path)
    parser.add_argument("--output", default="data/manifests/speechocean762_reading_analysis.csv", type=Path)
    parser.add_argument("--cmudict-dir", default="external_datasets/cmudict", type=Path)
    parser.add_argument("--reference-col", default="auto")
    parser.add_argument("--hypothesis-col", default="normalized_transcript")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_file(args.input, args.output, args.cmudict_dir, args.reference_col, args.hypothesis_col, args.limit)


if __name__ == "__main__":
    main()

