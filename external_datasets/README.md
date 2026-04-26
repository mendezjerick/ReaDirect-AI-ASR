# External Datasets

This directory is for public or third-party resources used by ReaDirect AI/ASR experiments.

## Active

- `cmudict/`: CMU Pronouncing Dictionary files used for word-to-phoneme enrichment.
- `speechocean762/`: active public pronunciation-scoring dataset for AI Phase 3 and later baseline ASR evaluation.

## Research-Only / Optional

- L2-ARCTIC is research-only optional and non-commercial only. It is not active in the current pipeline and must not be used for deployable model training, production inference, or commercial/government-client evaluation unless licensing changes.
- PF-STAR remains future optional only because it requires access/request.

Do not commit large public datasets, speech audio, private learner audio, archives, checkpoints, API keys, or `.env` files.
