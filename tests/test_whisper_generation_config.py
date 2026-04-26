from types import SimpleNamespace

from readirect_asr.finetuning.whisper_generation_config import prepare_whisper_generation_config


class FakeTokenizer:
    lang_to_id = {"en": 1, "<|en|>": 1}
    task_to_id = {"transcribe": 2, "<|transcribe|>": 2}

    def get_decoder_prompt_ids(self, language=None, task=None):
        ids = []
        if language:
            ids.append([1, self.lang_to_id.get(language, 1)])
        if task:
            ids.append([2, self.task_to_id.get(task, 2)])
        return ids


def test_generation_config_helper_patches_fake_model():
    model = SimpleNamespace(config=SimpleNamespace(), generation_config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    summary = prepare_whisper_generation_config(model, processor, language="en", task="transcribe", verbose=False)
    assert summary["has_lang_to_id"] is True
    assert model.generation_config.language == "en"
    assert model.generation_config.task == "transcribe"
    assert model.config.forced_decoder_ids == model.generation_config.forced_decoder_ids


def test_generation_config_helper_handles_minimal_english_only_model():
    model = SimpleNamespace(config=SimpleNamespace(), generation_config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    summary = prepare_whisper_generation_config(model, processor, language="en", task="transcribe", verbose=False)
    assert summary["language_set"] is None
    assert model.generation_config.language is None
    assert model.generation_config.forced_decoder_ids is None
