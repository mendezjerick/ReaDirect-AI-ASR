# Wav2Vec2 Delta

Delta continues from the final validated Beta model. It does not use Gamma as a
starting checkpoint and does not reset the tokenizer or CTC head.

## Effective Training Mix

The default virtual epoch contains 40,000 rows:

- GigaSpeech S: 20,000 effective rows (50%)
- SLR83 Southern English: 16,000 effective rows (40%)
- ReaDirect isolated letters: 4,000 effective rows (10%)
- SpeechOcean762: 0 training rows

All 8,492 indexed SLR83 male and female rows are included before deterministic
repetition. SLR83 labels are read from each folder's `line_index.csv`; labels
are never inferred from filenames.

The lightweight mix check reads GigaSpeech parquet metadata and does not
materialize the full dataset or rebuild its Arrow cache.

Training samples the configured 20,000 GigaSpeech rows directly from the local
parquet shards. Its smaller sampled cache is kept under
`reports/asr/delta/dataset_cache/`; it does not rebuild the old 230,068-row
Hugging Face parquet cache.

## Paths

```powershell
$env:BETA_CHECKPOINT_PATH = "models/asr/beta"
$env:GIGASPEECH_S_PATH = "external_datasets/gigaspeech_s_parquet/parquet-data/s"
$env:GIGASPEECH_CACHE_DIR = "external_datasets/hf_cache"
$env:SLR83_ROOT = "external_datasets/SLR83"
$env:READIRECT_LETTERS_ROOT = "external_datasets/readirect_letters"
$env:DELTA_EFFECTIVE_EPOCH_ROWS = "40000"
```

## Manual Commands

Verify SLR83 parsing and one 16 kHz decode:

```powershell
python scripts/validate_slr83_delta.py --config configs/wav2vec2_delta.yaml --decode-sample
```

Build and check the lightweight dataset mix plan:

```powershell
python scripts/build_delta_dataset_mix.py --config configs/wav2vec2_delta.yaml
```

Start training manually:

```powershell
python scripts/train_wav2vec2_delta.py --config configs/wav2vec2_delta.yaml
```

Run evaluation manually:

```powershell
python scripts/evaluate_wav2vec2_delta.py --config configs/wav2vec2_delta.yaml --model models/asr/delta
```

## Evaluation

Checkpoint selection uses the same 510-row shared validation set as Beta and
Gamma: 250 SpeechOcean rows plus 260 ReaDirect isolated-letter rows.
The final evaluation additionally reports GigaSpeech validation WER/CER when
the cached validation split is present. If that cache was removed during
storage cleanup, evaluation continues on the historical 510-row shared set
without downloading anything and marks GigaSpeech validation unavailable.

The output is:

```text
reports/asr/delta/delta_shared_validation_evaluation.json
```

## Output Isolation

```text
models/asr/delta/
models/asr/delta/checkpoints/
reports/asr/delta/
reports/asr/delta/logs/
```
