# Whisper Training Skeleton

This folder contains guarded training code. Training does not start unless you explicitly run `training/train_whisper.py` with `--run`.

Use this only to preview the planned configuration:

```powershell
python training/train_whisper.py --config configs/whisper_finetune_config.yaml --dry-run
```

Do not commit checkpoints, model weights, optimizer states, or generated trainer logs.
