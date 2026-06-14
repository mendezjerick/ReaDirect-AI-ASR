# Wav2Vec2 Epsilon

Epsilon is a controlled one-epoch refinement of the final Delta model. It does
not use Gamma or LibriSpeech for training.

## Training Mix

The default virtual epoch contains 40,000 rows:

- GigaSpeech S: 18,000 (45%)
- SLR83 Southern English train split: 12,000 (30%)
- SpeechOcean: 6,000 (15%)
- ReaDirect isolated letters: 4,000 (10%)

Sampling is deterministic. Sources smaller than their effective count are
repeated with a seeded shuffle. The raw row count does not determine the mix.

## SLR83 Split

SLR83 female and male Southern English folders are parsed through
`line_index.csv`; labels are never inferred from filenames. Seed 47 creates a
speaker-aware split separately across the two gender folders:

- 6,710 train rows from 45 speakers
- 1,782 held-out evaluation rows from 12 speakers
- zero speaker and source-ID overlap

The exact assignment is saved to:

```text
reports/asr/epsilon/slr83_split_manifest.json
```

Held-out rows are excluded from training, virtual repetition, validation,
early stopping, and checkpoint selection.

## GigaSpeech Filtering

The metadata scan rejects missing audio payloads, empty transcripts, duration
outside 0.3-25 seconds, unsupported/noise tags, and transcripts that normalize
to empty. The current local GigaSpeech S parquet was already curated: all
230,068 rows passed these metadata checks. Actual audio decoding still occurs
during preprocessing and requires working FFmpeg/TorchCodec support.

## Training

- Start model: `models/asr/delta`
- Output model/checkpoints: `models/asr/epsilon/`
- Reports/logs: `reports/asr/epsilon/`
- Default learning rate: `5e-6`
- Default epochs: `1`
- Maximum epochs: `2`
- Best model selection: shared validation WER
- Monitored metrics: WER, CER, isolated-letter accuracy

CLI overrides:

```powershell
--learning-rate 0.000005 --epochs 1
```

The tokenizer, 32-output CTC head, word delimiter, and 16 kHz processor must
match Delta and base-960h. Training refuses incompatible artifacts.

## Evaluation Policy

Shared evaluation uses the same 250 SpeechOcean and 260 ReaDirect-letter rows
used for Beta and Delta decoder comparisons. Epsilon supports greedy, no-LM
beam, and KenLM beam decoding.

The held-out SLR83 evaluation is separate and reporting-only. For deployment,
LM beam can be selected for open sentence ASR, but isolated letters should use
no-LM beam or a closed-set A-Z decoder.

## Manual Commands

All commands use the repository virtual environment directly and do not require
activation. See the implementation completion message for the ordered commands
to validate, train, evaluate all decoder modes, and generate the comparison.
