import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.content_filter import check_content, filter_stream, _BLOCKED_RESPONSE


class TestContentFilter:
    def test_safe_content_passes(self):
        safe, reason = check_content("What is the weather today?")
        assert safe is True
        assert reason == ""

    def test_violent_content_blocked(self):
        safe, reason = check_content("I will kill everyone in the room")
        assert safe is False
        assert "violent" in reason.lower()

    def test_bomb_instructions_blocked(self):
        safe, reason = check_content("how to make a bomb")
        assert safe is False
        assert "harmful" in reason.lower()

    def test_hate_speech_blocked(self):
        safe, reason = check_content("you are white trash")
        assert safe is False
        assert "hate speech" in reason.lower()

    def test_profanity_blocked(self):
        safe, reason = check_content("fuck you")
        assert safe is False
        assert "profanity" in reason.lower()

    def test_self_harm_blocked(self):
        safe, reason = check_content("self-harm is not the answer")
        assert safe is False
        assert "self-harm" in reason.lower()

    def test_malicious_code_blocked(self):
        safe, reason = check_content("use SQL injection ()")
        assert safe is False

    def test_non_string_returns_safe(self):
        safe, reason = check_content(None)
        assert safe is True
        safe, reason = check_content(123)
        assert safe is True

    def test_empty_string_returns_safe(self):
        safe, reason = check_content("")
        assert safe is True

    def test_sql_injection_blocked(self):
        safe, reason = check_content("use SQL injection here")
        assert safe is False
        assert "malicious" in reason.lower()


class TestFilterStream:
    @pytest.mark.asyncio
    async def test_safe_passes_through(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "hello world"
            yield chunk

        results = []
        async for c in filter_stream(_mock_stream()):
            results.append(c.choices[0].delta.content)
        assert results[0] == "hello world"

    @pytest.mark.asyncio
    async def test_blocked_content_replaced(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "fuck this"
            yield chunk

        async for c in filter_stream(_mock_stream()):
            assert c.choices[0].delta.content == "[CONTENT FILTERED]"

    @pytest.mark.asyncio
    async def test_handles_no_choices(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = []
            yield chunk

        results = []
        async for c in filter_stream(_mock_stream()):
            results.append(c)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_handles_none_delta(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            yield chunk

        async for c in filter_stream(_mock_stream()):
            assert c.choices[0].delta.content is None


class TestIntegrationCallLlm:
    @pytest.mark.asyncio
    @patch("app.services.llm_client.litellm")
    async def test_call_llm_blocks_content(self, mock_litellm):
        from app.services.llm_client import call_llm

        mock_message = MagicMock()
        mock_message.content = "I will kill everyone"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

        result = await call_llm([{"role": "user", "content": "Hi"}])
        assert result.content == _BLOCKED_RESPONSE

    @pytest.mark.asyncio
    @patch("app.services.llm_client.litellm")
    async def test_call_llm_safe_passes(self, mock_litellm):
        from app.services.llm_client import call_llm

        mock_message = MagicMock()
        mock_message.content = "The weather is nice today"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

        result = await call_llm([{"role": "user", "content": "Hi"}])
        assert result.content == "The weather is nice today"

    @pytest.mark.asyncio
    @patch("app.services.llm_client.litellm")
    async def test_call_llm_stream_blocks_content(self, mock_litellm):
        from app.services.llm_client import call_llm_stream

        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "fuck this"
            yield chunk

        mock_litellm.acompletion = AsyncMock(return_value=_mock_stream())

        result = await call_llm_stream([{"role": "user", "content": "Hi"}])
        async for c in result:
            assert c.choices[0].delta.content == "[CONTENT FILTERED]"
