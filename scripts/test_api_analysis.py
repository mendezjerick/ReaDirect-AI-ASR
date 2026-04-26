from __future__ import annotations

import argparse
import json
from typing import Any

import httpx


def parse_accepted(values: list[str]) -> list[str]:
    answers: list[str] = []
    for value in values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                answers.append(cleaned)
    return answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the ReaDirect AI analysis API.")
    parser.add_argument("--mode", choices=["text", "audio"], default="text")
    parser.add_argument("--expected-text", default="")
    parser.add_argument("--actual-text", default="")
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--accepted-answer", action="append", default=[])
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = "/analyze-text" if args.mode == "text" else "/analyze-audio"
    payload: dict[str, Any] = {
        "expected_text": args.expected_text or None,
        "accepted_answers": parse_accepted(args.accepted_answer),
        "prompt_id": args.prompt_id,
        "debug": args.debug,
    }
    if args.mode == "text":
        payload["actual_text"] = args.actual_text
    else:
        payload["audio_path"] = args.audio_path
    headers = {"X-ReaDirect-AI-Token": args.token} if args.token else {}
    try:
        response = httpx.post(f"{args.base_url.rstrip('/')}{endpoint}", json=payload, headers=headers, timeout=60)
    except httpx.ConnectError:
        print("API server is not running. Start it with:")
        print("uvicorn api.main:app --reload --port 8001")
        return
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()

