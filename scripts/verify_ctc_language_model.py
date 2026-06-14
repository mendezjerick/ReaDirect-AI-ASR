from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.ctc_decoding import CTCTextDecoder, DecodeSettings
from training.wav2vec2_manifest_utils import resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and validate a KenLM model against the current Wav2Vec2 tokenizer."
    )
    parser.add_argument("--lm_path", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/asr/delta"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    from transformers import Wav2Vec2Processor

    model_path = resolve_repo_path(args.model)
    lm_path = resolve_repo_path(args.lm_path)
    processor = Wav2Vec2Processor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    decoder = CTCTextDecoder(
        processor,
        DecodeSettings(
            decode_mode="beam_lm",
            beam_width=100,
            alpha=args.alpha,
            beta=args.beta,
            lm_path=str(lm_path),
        ),
    )
    metadata = decoder.metadata()
    if metadata["language_model_used"] is not True:
        raise RuntimeError("KenLM validation failed: language_model_used is not true.")
    print("LANGUAGE MODEL BEAM SEARCH ACTIVE: True")
    print(f"Loaded KenLM language model: {lm_path}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
