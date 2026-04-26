# Speechocean762

Speechocean762 is the active public speech/pronunciation dataset for AI Phase 3.

Expected local layout:

```text
external_datasets/speechocean762/
├── raw/
│   └── speechocean762.tar.gz
├── extracted/
└── processed/
```

Commands:

```powershell
python scripts/inspect_speechocean762.py --dataset-dir external_datasets/speechocean762 --print-tree
python scripts/extract_speechocean762.py --archive external_datasets/speechocean762/raw/speechocean762.tar.gz --dest external_datasets/speechocean762/extracted
python scripts/build_speechocean762_manifest.py --dataset-dir external_datasets/speechocean762/extracted --cmudict-dir external_datasets/cmudict --output data/manifests/speechocean762_manifest.csv
```

Do not commit downloaded audio, archives, extracted dataset files, processed dataset files, or generated manifests.
