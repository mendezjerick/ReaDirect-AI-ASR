from pathlib import Path

import yaml


def test_whisper_config_loads_with_conservative_rtx3060_settings():
    config = yaml.safe_load(Path("configs/whisper_finetune_config.yaml").read_text(encoding="utf-8"))
    assert config["model"]["name_or_path"] == "openai/whisper-base.en"
    assert config["training"]["output_dir"].startswith("model_artifacts/")
    assert config["runtime"]["dry_run_default"] is True
    assert config["training"]["per_device_train_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["fp16"] is True
    assert config["training"]["gradient_checkpointing"] is True
    assert config["data"]["audio_loading_backend"] == "librosa"
    assert config["training"]["evaluation_strategy"] == "no"
    assert config["training"]["run_eval_during_training"] is False
