from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.wav2vec2_manifest_utils import audio_info, make_manifest_row, resolve_repo_path, write_jsonl


LETTERS = list(string.ascii_uppercase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare letter and word evaluation manifests from local audio datasets.")
    parser.add_argument("--mswc-root", default="external_datasets/audio/extracted/mswc_microset/mswc_microset/en", type=Path)
    parser.add_argument("--letter-root", default="external_datasets/audio/extracted/english_audio", type=Path)
    parser.add_argument("--processed-letter-dir", default="external_datasets/audio/processed/english_letters", type=Path)
    parser.add_argument("--mswc-split", choices=("dev", "test", "train"), default="test")
    parser.add_argument("--max-per-word", type=int, default=None)
    parser.add_argument("--output-words", default="external_datasets/manifests/mswc_words_test.jsonl", type=Path)
    parser.add_argument("--output-letters", default="external_datasets/manifests/english_letters_eval.jsonl", type=Path)
    parser.add_argument("--output-combined", default="external_datasets/manifests/readirect_letters_words_audio_eval.jsonl", type=Path)
    parser.add_argument("--report", default="external_datasets/manifests/readirect_letters_words_audio_eval_report.md", type=Path)
    return parser.parse_args()


def prepare_mswc_words(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = resolve_repo_path(args.mswc_root)
    csv_path = root / f"en_{args.mswc_split}.csv"
    if not csv_path.exists():
        return [], {"error": f"missing {csv_path}", "word_counts": {}}

    rows: list[dict[str, Any]] = []
    word_counts: Counter[str] = Counter()
    skipped_missing_audio = 0
    skipped_invalid = 0
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for item in reader:
            word = str(item.get("WORD", "")).strip().lower()
            valid = str(item.get("VALID", "")).strip().lower() == "true"
            link = str(item.get("LINK", "")).strip().replace("\\", "/")
            if not word or not valid or not link:
                skipped_invalid += 1
                continue
            if args.max_per_word is not None and word_counts[word] >= args.max_per_word:
                continue
            audio_path = root / "clips" / link
            if not audio_path.exists():
                skipped_missing_audio += 1
                continue
            row = make_manifest_row(
                audio_path=audio_path,
                text=word,
                dataset="mswc",
                split=args.mswc_split,
                speaker_id=str(item.get("SPEAKER", "")),
                source_id=Path(link).stem,
                metadata={
                    "source_dataset": "Multilingual Spoken Words Corpus microset",
                    "original_link": link,
                    "gender": item.get("GENDER", ""),
                    "prompt_type": "word",
                },
            )
            row["prompt_type"] = "word"
            rows.append(row)
            word_counts[word] += 1
    return rows, {
        "csv_path": str(csv_path),
        "word_counts": dict(word_counts),
        "skipped_missing_audio": skipped_missing_audio,
        "skipped_invalid": skipped_invalid,
    }


def best_letter_intervals(audio_path: Path) -> tuple[Any | None, int | None, list[tuple[int, int]], str]:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    best: list[tuple[int, int]] = []
    best_label = "none"
    for top_db in (35, 30, 25, 20):
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=160)
        filtered = [
            (int(start), int(end))
            for start, end in intervals
            if 0.12 <= (end - start) / sr <= 1.25
        ]
        if len(filtered) == 26:
            return y, sr, filtered, f"top_db={top_db}"
        if not best or abs(len(filtered) - 26) < abs(len(best) - 26):
            best = filtered
            best_label = f"top_db={top_db};count={len(filtered)}"
    return y, sr, [], f"no_exact_26;best={best_label}"


def prepare_letter_segments(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    letter_root = resolve_repo_path(args.letter_root)
    processed_dir = resolve_repo_path(args.processed_letter_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    files_used = 0
    files_skipped: list[dict[str, Any]] = []
    for wav_path in sorted(letter_root.rglob("*.wav")):
        y, sr, intervals, method = best_letter_intervals(wav_path)
        if y is None or sr is None or len(intervals) != 26:
            files_skipped.append({"file": str(wav_path), "reason": method})
            continue
        files_used += 1
        stem = wav_path.stem
        for index, letter in enumerate(LETTERS):
            start, end = intervals[index]
            pad = int(0.06 * sr)
            clip_start = max(0, start - pad)
            clip_end = min(len(y), end + pad)
            output_audio = processed_dir / stem / f"{index + 1:02d}_{letter}.wav"
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_audio), y[clip_start:clip_end], sr)
            row = make_manifest_row(
                audio_path=output_audio,
                text=letter,
                dataset="english_audio_letters",
                split="test",
                speaker_id=stem,
                source_id=f"{stem}_{letter}_{index + 1:02d}",
                metadata={
                    "source_audio": str(wav_path),
                    "segmentation_method": method,
                    "prompt_type": "letter",
                    "segment_index": index + 1,
                },
            )
            row["prompt_type"] = "letter"
            rows.append(row)
    return rows, {
        "files_seen": len(list(letter_root.rglob("*.wav"))),
        "files_used": files_used,
        "files_skipped": files_skipped,
        "processed_dir": str(processed_dir),
    }


def write_report(path: Path, letters: list[dict[str, Any]], words: list[dict[str, Any]], letter_report: dict[str, Any], word_report: dict[str, Any]) -> None:
    output = resolve_repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    word_counts = Counter(row["text"] for row in words)
    letter_counts = Counter(row["text"] for row in letters)
    lines = [
        "# Audio Letter/Word Evaluation Manifest Report",
        "",
        f"- Letter rows: {len(letters)}",
        f"- Word rows: {len(words)}",
        f"- Combined rows: {len(letters) + len(words)}",
        f"- Letter source WAV files seen: {letter_report.get('files_seen', 0)}",
        f"- Letter source WAV files used: {letter_report.get('files_used', 0)}",
        f"- Processed letter clips: `{letter_report.get('processed_dir', '')}`",
        f"- MSWC CSV: `{word_report.get('csv_path', '')}`",
        f"- MSWC skipped missing audio: {word_report.get('skipped_missing_audio', 0)}",
        f"- MSWC skipped invalid rows: {word_report.get('skipped_invalid', 0)}",
        "",
        "## Letter Counts",
        "",
    ]
    for letter in LETTERS:
        lines.append(f"- {letter}: {letter_counts.get(letter, 0)}")
    lines.extend(["", "## Word Counts", ""])
    for word, count in sorted(word_counts.items()):
        lines.append(f"- {word}: {count}")
    lines.extend(["", "## Skipped Letter Files", ""])
    skipped = letter_report.get("files_skipped", [])
    if skipped:
        for item in skipped:
            lines.append(f"- `{item['file']}`: {item['reason']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Letter clips are generated by silence segmentation of alphabet-recitation WAV files.",
            "- Only source files that segment into exactly 26 usable intervals are included.",
            "- MSWC word clips use the dataset's English test CSV by default.",
            "- These manifests are for evaluation only; no training data or model weights are modified.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    words, word_report = prepare_mswc_words(args)
    letters, letter_report = prepare_letter_segments(args)
    combined = letters + words

    word_count = write_jsonl(args.output_words, words)
    letter_count = write_jsonl(args.output_letters, letters)
    combined_count = write_jsonl(args.output_combined, combined)
    write_report(args.report, letters, words, letter_report, word_report)

    print(f"Wrote {word_count} word rows to {args.output_words}")
    print(f"Wrote {letter_count} letter rows to {args.output_letters}")
    print(f"Wrote {combined_count} combined rows to {args.output_combined}")
    print(f"Report: {args.report}")
    print(json.dumps({"letters": letter_count, "words": word_count, "combined": combined_count}, indent=2))
    return 0 if combined_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
