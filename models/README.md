# Models

Local downloaded models, checkpoints, and experiment outputs may be placed here during development.

Runtime ASR expects:

```text
models/wav2vec2-readirect-asr/
models/wav2vec2-phoneme/
models/wav2vec2-base-960h/
```

`wav2vec2-readirect-asr` is the primary model. `wav2vec2-phoneme` is supporting evidence. `wav2vec2-base-960h` is used only when `ALLOW_WAV2VEC2_BASE_FALLBACK=true`.

Do not commit model files, checkpoints, fine-tuned weights, or large artifacts. This repository tracks only this README and `.gitkeep`.
