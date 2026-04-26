from training.train_whisper import build_seq2seq_training_arguments_kwargs


class EvalStrategyArgs:
    def __init__(self, output_dir=None, eval_strategy=None, save_strategy=None, predict_with_generate=None, max_steps=None):
        pass


class EvaluationStrategyArgs:
    def __init__(self, output_dir=None, evaluation_strategy=None, save_strategy=None, predict_with_generate=None, max_steps=None):
        pass


def test_training_arguments_uses_eval_strategy_when_supported():
    kwargs = build_seq2seq_training_arguments_kwargs(
        EvalStrategyArgs,
        {"output_dir": "out", "evaluation_strategy": "no", "save_strategy": "steps", "predict_with_generate": True, "max_steps": 1},
        run_eval_during_training=False,
    )
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["predict_with_generate"] is False
    assert "evaluation_strategy" not in kwargs


def test_training_arguments_uses_evaluation_strategy_when_supported():
    kwargs = build_seq2seq_training_arguments_kwargs(
        EvaluationStrategyArgs,
        {"output_dir": "out", "evaluation_strategy": "steps", "save_strategy": "steps", "predict_with_generate": True, "max_steps": 1},
        run_eval_during_training=True,
    )
    assert kwargs["evaluation_strategy"] == "steps"
    assert kwargs["predict_with_generate"] is True
    assert "eval_strategy" not in kwargs
