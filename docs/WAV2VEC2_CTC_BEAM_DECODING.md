# Wav2Vec2 CTC Beam Decoding

This is a decoder-only experiment. It does not modify model weights,
tokenizers, CTC heads, or training behavior.

## Decoder Modes

- `greedy`: existing argmax plus `Wav2Vec2Processor.batch_decode`
- `beam`: true CTC prefix beam search

Beam mode prefers `pyctcdecode`. If it is unavailable and no external language
model is requested, evaluation uses the included pure-Python CTC prefix beam
search. It never silently falls back to greedy.

The tokenizer mapping is:

- `<pad>` ID 0: CTC blank
- `|` ID 4: word delimiter, converted to a space
- BOS, EOS, and UNK IDs: suppressed before beam decoding
- letter and apostrophe IDs: normal emitted CTC tokens

Without an external language model, `alpha` has no scoring effect. `beta`
remains a word-boundary bonus in the pure-Python decoder. External KenLM
support requires `pyctcdecode` and a compatible `--lm_path`.

## Install Optional Backend

```powershell
python -m pip install --no-deps pyctcdecode pygtrie
```

`pyctcdecode 0.5.0` declares an old NumPy constraint. Using `--no-deps`
preserves the existing NumPy 2.x environment; this combination was verified
with the current tokenizer. If the backend still fails to initialize, the
script prints the failure and uses true pure-Python prefix beam search without
an LM.

## Manual Evaluations

```powershell
python scripts/evaluate_wav2vec2_decoder.py --model-name beta --decode_mode greedy
python scripts/evaluate_wav2vec2_decoder.py --model-name beta --decode_mode beam --beam_width 50 --alpha 0.5 --beta 1.0
python scripts/evaluate_wav2vec2_decoder.py --model-name delta --decode_mode greedy
python scripts/evaluate_wav2vec2_decoder.py --model-name delta --decode_mode beam --beam_width 50 --alpha 0.5 --beta 1.0
python scripts/compare_wav2vec2_decoders.py
```

Outputs are saved under:

```text
reports/asr/decoder_comparison/
```
