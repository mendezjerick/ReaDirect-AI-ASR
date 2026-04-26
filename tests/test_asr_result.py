from readirect_asr.asr.result import ASRResult, ASRSegment, ASRWord


def test_asr_result_serializes_segments_and_words() -> None:
    result = ASRResult(
        transcript="hello",
        normalized_transcript="hello",
        language="en",
        segments=[ASRSegment(start=0.0, end=1.0, text="hello")],
        words=[ASRWord(word="hello", start=0.0, end=1.0, probability=0.9)],
        provider="mock",
        model_size="test",
    )

    data = result.to_dict()

    assert data["segments"][0]["text"] == "hello"
    assert data["words"][0]["probability"] == 0.9

