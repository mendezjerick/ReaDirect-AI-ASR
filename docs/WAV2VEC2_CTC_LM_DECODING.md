# Wav2Vec2 CTC Beam Search With KenLM

This is a decoder-only experiment. It does not change model weights,
tokenizers, CTC heads, checkpoints, or training behavior.

## Modes

- `greedy`: existing Transformers CTC argmax decoding
- `beam`: true CTC beam search without an external language model
- `beam_lm`: `pyctcdecode` beam search with a required KenLM `.arpa` or `.bin`

`beam_lm` is strict. A missing, empty, invalid, or unloadable LM causes an
error. It never falls back to no-LM beam or greedy decoding.

The decoder builds its alphabet from the model's current tokenizer:

- `<pad>` is the CTC blank and maps to an empty label.
- `|` is the word delimiter and maps to a space.
- BOS, EOS, and UNK logit columns are suppressed.
- Tokenizer IDs must be contiguous and match the model logit vocabulary.

## Dependencies

On Windows, KenLM must be compiled locally. Install Microsoft Visual Studio
2022 Build Tools with the C++ workload first:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements --accept-source-agreements
```

Close and reopen PowerShell after that installation. Then install:

```powershell
python -m pip install --no-deps pyctcdecode pygtrie
python -m pip install https://github.com/kpu/kenlm/archive/master.zip
```

KenLM's isolated build environment installs CMake automatically. Messages
about missing `bash` may appear during metadata generation on Windows; the
blocking failure is a missing C/C++ compiler. The verification command below
must report `language_model_used: true`.

## LM Location

Place an existing English KenLM model at either location:

```text
external_datasets/language_models/english_3gram.arpa
external_datasets/language_models/english_3gram.bin
```

This task does not train or download a language model. Both formats are passed
directly to `pyctcdecode`.

## Output Files

All outputs are stored under:

```text
reports/asr/decoder_lm_comparison/
```

The final comparison contains greedy, no-LM beam, and LM beam results for Beta
and Delta. Existing greedy and no-LM reports are imported from
`reports/asr/decoder_comparison/` after their decode modes are validated.

The LM sweep runs Delta with:

- alpha: `0.3`, `0.5`, `0.7`
- beta: `0.5`, `1.0`, `1.5`
- beam width: `100`

It selects the lowest shared WER, then the lowest shared CER.

## Hotwords

Hotwords are optional:

```powershell
--hotwords READIRECT ALPHA BETA DELTA --hotword_weight 5.0
```

Do not use hotwords for the baseline LM comparison unless the same list is
used consistently. KenLM is not expected to improve isolated-letter accuracy;
a later letter-specific path should use closed-set A-Z decoding.
