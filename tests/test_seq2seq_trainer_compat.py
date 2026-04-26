from training.train_whisper import build_seq2seq_trainer_kwargs


class NewTrainer:
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None, compute_metrics=None):
        pass


class OldTrainer:
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, compute_metrics=None):
        pass


class MinimalTrainer:
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, compute_metrics=None):
        pass


class Processor:
    tokenizer = "tokenizer-object"


def _kwargs(trainer_cls):
    return build_seq2seq_trainer_kwargs(
        trainer_cls=trainer_cls,
        processor=Processor(),
        args="args",
        model="model",
        train_dataset="train",
        eval_dataset="eval",
        data_collator="collator",
        compute_metrics="metrics",
    )


def test_new_transformers_uses_processing_class():
    kwargs = _kwargs(NewTrainer)
    assert "processing_class" in kwargs
    assert "tokenizer" not in kwargs


def test_old_transformers_uses_tokenizer():
    kwargs = _kwargs(OldTrainer)
    assert kwargs["tokenizer"] == "tokenizer-object"
    assert "processing_class" not in kwargs


def test_minimal_trainer_gets_neither_processing_class_nor_tokenizer():
    kwargs = _kwargs(MinimalTrainer)
    assert "processing_class" not in kwargs
    assert "tokenizer" not in kwargs
