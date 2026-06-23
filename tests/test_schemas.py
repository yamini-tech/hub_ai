import pytest
from pydantic import ValidationError
from app.schemas import (
    AIRequest, SummarizeRequest, ParseRequest,
    SentimentResponse, TaskType, GatewayRequest,
)


class TestAIRequest:
    def test_valid_request(self):
        req = AIRequest(text="Hello")
        assert req.text == "Hello"
        assert req.session_id == "default_session"
        assert req.task_type == "chat"

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            AIRequest(text="")

    def test_text_too_long_fails(self):
        with pytest.raises(ValidationError):
            AIRequest(text="x" * 50001)

    def test_custom_session_id(self):
        req = AIRequest(text="Hi", session_id="custom")
        assert req.session_id == "custom"

    def test_custom_task_type(self):
        req = AIRequest(text="Hi", task_type="analyze")
        assert req.task_type == "analyze"


class TestSummarizeRequest:
    def test_valid(self):
        req = SummarizeRequest(text="Summarize this")
        assert req.text == "Summarize this"

    def test_empty_fails(self):
        with pytest.raises(ValidationError):
            SummarizeRequest(text="")

    def test_too_long_fails(self):
        with pytest.raises(ValidationError):
            SummarizeRequest(text="x" * 100001)


class TestParseRequest:
    def test_valid(self):
        req = ParseRequest(text="Parse this")
        assert req.text == "Parse this"
        assert req.schema_hint is None

    def test_with_schema_hint(self):
        req = ParseRequest(text="Parse this", schema_hint="JSON output")
        assert req.schema_hint == "JSON output"

    def test_schema_hint_too_long_fails(self):
        with pytest.raises(ValidationError):
            ParseRequest(text="Parse this", schema_hint="x" * 501)

    def test_empty_fails(self):
        with pytest.raises(ValidationError):
            ParseRequest(text="")


class TestSentimentResponse:
    def test_valid_positive(self):
        resp = SentimentResponse(
            sentiment="positive", confidence=0.95, key_phrases=["good", "great"]
        )
        assert resp.sentiment == "positive"

    def test_valid_negative(self):
        resp = SentimentResponse(
            sentiment="negative", confidence=0.8, key_phrases=["bad"]
        )
        assert resp.sentiment == "negative"

    def test_valid_neutral(self):
        resp = SentimentResponse(
            sentiment="neutral", confidence=0.5, key_phrases=["ok"]
        )
        assert resp.sentiment == "neutral"

    def test_invalid_sentiment_fails(self):
        with pytest.raises(ValidationError):
            SentimentResponse(
                sentiment="unknown", confidence=0.5, key_phrases=["ok"]
            )

    def test_confidence_too_low_fails(self):
        with pytest.raises(ValidationError):
            SentimentResponse(
                sentiment="positive", confidence=-0.1, key_phrases=["ok"]
            )

    def test_confidence_too_high_fails(self):
        with pytest.raises(ValidationError):
            SentimentResponse(
                sentiment="positive", confidence=1.5, key_phrases=["ok"]
            )

    def test_too_many_key_phrases_fails(self):
        with pytest.raises(ValidationError):
            SentimentResponse(
                sentiment="positive", confidence=0.5,
                key_phrases=["a"] * 21,
            )


class TestTaskType:
    def test_values(self):
        assert TaskType.SUMMARIZE == "summarize"
        assert TaskType.PARSE == "parse"
        assert TaskType.AGENT == "agent"
        assert TaskType.PROCESS == "process"


class TestGatewayRequest:
    def test_valid_stream_default(self):
        req = GatewayRequest(task_type=TaskType.SUMMARIZE, text="Summarize this")
        assert req.stream is True

    def test_valid_stream_false(self):
        req = GatewayRequest(
            task_type=TaskType.PARSE, text="Parse this", stream=False
        )
        assert req.stream is False

    def test_with_schema_hint(self):
        req = GatewayRequest(
            task_type=TaskType.PARSE, text="Parse this",
            schema_hint="JSON", stream=False,
        )
        assert req.schema_hint == "JSON"

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            GatewayRequest(task_type=TaskType.SUMMARIZE, text="")

    def test_text_too_long_fails(self):
        with pytest.raises(ValidationError):
            GatewayRequest(task_type=TaskType.SUMMARIZE, text="x" * 50001)

    def test_custom_session_id(self):
        req = GatewayRequest(
            task_type=TaskType.AGENT, text="Hi", session_id="my_session"
        )
        assert req.session_id == "my_session"

    def test_all_task_types(self):
        for tt in TaskType:
            req = GatewayRequest(task_type=tt, text="test")
            assert req.task_type == tt
