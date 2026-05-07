# Models

Local downloaded models, checkpoints, and experiment outputs may be placed here during development.

Runtime ASR expects:

```text
models/wav2vec2-readirect-asr-letters-v2/
models/wav2vec2-readirect-asr/
models/wav2vec2-phoneme/
```

`wav2vec2-readirect-asr-letters-v2` is the primary active model. `wav2vec2-readirect-asr` is the previous v1 model retained for fallback/reference. `wav2vec2-phoneme` is supporting evidence. Fallback is used only when `ALLOW_WAV2VEC2_BASE_FALLBACK=true`.

Do not commit model files, checkpoints, fine-tuned weights, or large artifacts. This repository tracks only this README and `.gitkeep`.
