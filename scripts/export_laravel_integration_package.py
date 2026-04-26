from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DOCS = [
    "LARAVEL_INTEGRATION_CONTRACT.md",
    "FASTAPI_SERVICE.md",
    "MODEL_ARTIFACT_SHARING.md",
    "CONTENT_ENRICHMENT.md",
    "ADAPTIVE_TUTORING_ENGINE.md",
    "API_EXAMPLES.md",
    "MAIN_REPO_IMPORT_GUIDE.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Laravel integration docs and safe content artifacts.")
    parser.add_argument("--output-dir", default="exports/readirect-ai-laravel-integration", type=Path)
    parser.add_argument("--zip-output", default="exports/readirect-ai-laravel-integration.zip", type=Path)
    parser.add_argument("--include-model-artifact", action="store_true")
    parser.add_argument("--include-enriched-content", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    included: list[str] = []
    excluded = [
        "Speechocean762 archive and extracted audio",
        "external training datasets",
        "training JSONL files",
        "training manifests",
        "training checkpoints",
        "real learner audio",
        ".env and API tokens",
    ]
    if args.dry_run:
        print(f"Would create export at {args.output_dir}")
        print("Would include docs, env example, API examples, deployment notes, and safe enriched content if present.")
        return 0
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    (args.output_dir / "docs").mkdir(parents=True)
    (args.output_dir / "env").mkdir()
    (args.output_dir / "examples").mkdir()
    (args.output_dir / "content").mkdir()
    (args.output_dir / "deployment").mkdir()

    for name in DOCS:
        source = Path("docs") / name
        if source.exists():
            shutil.copy2(source, args.output_dir / "docs" / name)
            included.append(f"docs/{name}")
    _write_laravel_env(args.output_dir / "env" / "readirect-laravel-ai.env.example")
    included.append("env/readirect-laravel-ai.env.example")
    _write_examples(args.output_dir / "examples")
    included.extend([f"examples/{path.name}" for path in (args.output_dir / "examples").iterdir()])
    _write_deployment_notes(args.output_dir / "deployment")
    included.extend([f"deployment/{path.name}" for path in (args.output_dir / "deployment").iterdir()])
    if args.include_enriched_content:
        _copy_if_exists(Path("content_bank_enriched/enriched_content_index.csv"), args.output_dir / "content" / "enriched_content_index.csv", included)
        _copy_if_exists(Path("content_bank_enriched/readirect-enriched-content.zip"), args.output_dir / "content" / "readirect-enriched-content.zip", included)
    _write_content_readme(args.output_dir / "content" / "README.md")
    included.append("content/README.md")
    if args.include_model_artifact:
        model_dir = Path("model_artifacts/readirect-whisper-base-en-v1-hf")
        if model_dir.exists():
            shutil.copytree(model_dir, args.output_dir / "model_artifacts" / model_dir.name)
            included.append(f"model_artifacts/{model_dir.name}/")
        else:
            excluded.append("model artifact requested but not found")
    else:
        excluded.append("model artifacts by default")
    _write_manifest(args.output_dir / "EXPORT_MANIFEST.md", included, excluded, args.include_model_artifact)
    args.zip_output.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_output.exists():
        args.zip_output.unlink()
    shutil.make_archive(str(args.zip_output.with_suffix("")), "zip", args.output_dir)
    print(f"Export folder: {args.output_dir}")
    print(f"Export ZIP: {args.zip_output}")
    return 0


def _copy_if_exists(source: Path, dest: Path, included: list[str]) -> None:
    if source.exists():
        shutil.copy2(source, dest)
        included.append(str(dest.relative_to(dest.parents[1])).replace("\\", "/"))


def _write_laravel_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "READIRECT_AI_ENABLED=true",
                "READIRECT_AI_BASE_URL=http://127.0.0.1:8001",
                "READIRECT_AI_API_TOKEN=",
                "READIRECT_AI_TIMEOUT_SECONDS=60",
                "READIRECT_AI_ANALYZE_AUDIO_ENDPOINT=/analyze-audio",
                "READIRECT_AI_ANALYZE_TEXT_ENDPOINT=/analyze-text",
                "READIRECT_AI_RECOMMEND_NEXT_ENDPOINT=/recommend-next",
                "READIRECT_AI_CONTENT_ITEM_ENDPOINT=/content-item",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_examples(folder: Path) -> None:
    examples = {
        "analyze_audio_request.json": {"audio_path": "data/samples/sample.wav", "expected_text": "cat", "accepted_answers": ["cat"], "prompt_id": "M2-001", "module_key": "module_2", "activity_type": "read_word", "learner_history": [], "debug": False},
        "analyze_audio_response.json": {"ok": True, "request_id": "uuid", "transcript": "cat", "normalized_transcript": "cat", "provider": "hf_whisper_local", "expected_text": "cat", "is_correct": True, "similarity_label": "exact", "error_type": "correct", "warnings": [], "error": None},
        "analyze_text_request.json": {"expected_text": "cat", "actual_text": "cap", "accepted_answers": ["cat"], "debug": True},
        "recommend_next_request.json": {"learner_history": [{"error_type": "final_sound_error", "skill_signal": "final_consonant", "is_correct": False}], "candidate_items": [{"prompt_id": "M2-014", "expected_text": "hat", "error_focus": "final_consonant", "difficulty_level": "easy"}]},
        "recommend_next_response.json": {"ok": True, "selected_item": {"prompt_id": "M2-014", "expected_text": "hat"}, "recommendation": {"primary_focus": "final_consonant", "recommended_action": "practice"}},
    }
    for name, data in examples.items():
        (folder / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_deployment_notes(folder: Path) -> None:
    (folder / "AI_SERVICE_RUNTIME_REQUIREMENTS.md").write_text("Runtime needs Python dependencies, CMUdict, content metadata, and the selected model artifact. Speechocean762 is not required at runtime.\n", encoding="utf-8")
    (folder / "START_AI_SERVICE.md").write_text("Run: `uvicorn api.main:app --host 127.0.0.1 --port 8001`\n", encoding="utf-8")
    (folder / "MODEL_PATHS.md").write_text("Place Hugging Face model at `ReaDirect-AI-ASR/model_artifacts/readirect-whisper-base-en-v1-hf/`. Optional CT2 model path: `model_artifacts/readirect-whisper-base-en-v1-ct2/`.\n", encoding="utf-8")


def _write_content_readme(path: Path) -> None:
    path.write_text(
        "Place reviewed enriched content ZIP in `ReaDirect/content-bank/import/readirect-enriched-content.zip` or extract reviewed CSVs to `ReaDirect/database/seed-data/readirect/enriched/`.\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, included: list[str], excluded: list[str], include_model: bool) -> None:
    lines = ["# Export Manifest", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "", "## Included", ""]
    lines.extend(f"- {item}" for item in included)
    lines.extend(["", "## Excluded", ""])
    lines.extend(f"- {item}" for item in excluded)
    lines.extend(["", f"Model artifact included: `{include_model}`", "External datasets included: `false`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
