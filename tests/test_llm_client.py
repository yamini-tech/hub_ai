import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException


class TestCallLlm:
    @patch("app.services.llm_client.litellm")
    async def test_call_llm_returns_message(self, mock_litellm):
        from app.services.llm_client import call_llm

        mock_message = MagicMock()
        mock_message.content = "Hello!"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

        result = await call_llm([{"role": "user", "content": "Hi"}])
        assert result.content == "Hello!"

    @patch("app.services.llm_client.litellm")
    async def test_call_llm_passes_model(self, mock_litellm):
        from app.services.llm_client import call_llm

        mock_message = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

        await call_llm([{"role": "user", "content": "Hi"}], model="custom-model")
        mock_litellm.acompletion.assert_called_with(
            model="custom-model",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None, response_format=None,
        )

    @patch("app.services.llm_client.litellm")
    async def test_call_llm_raises_on_error(self, mock_litellm):
        from app.services.llm_client import call_llm

        mock_litellm.acompletion = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(HTTPException) as exc:
            await call_llm([{"role": "user", "content": "Hi"}])
        assert exc.value.status_code == 500
        assert "API error" in exc.value.detail


class TestCallLlmStream:
    @patch("app.services.llm_client.litellm")
    async def test_call_llm_stream_returns_redacted_stream(self, mock_litellm):
        from app.services.llm_client import call_llm_stream

        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "contact me at user@example.com"
            yield chunk

        mock_litellm.acompletion = AsyncMock(return_value=_mock_stream())

        result = await call_llm_stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for c in result:
            chunks.append(c.choices[0].delta.content)
        assert "[EMAIL]" in chunks[0]
        assert "user@example.com" not in chunks[0]
        mock_litellm.acompletion.assert_called_with(
            model="ollama/llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None, stream=True,
        )

    @patch("app.services.llm_client.litellm")
    async def test_call_llm_stream_raises_on_error(self, mock_litellm):
        from app.services.llm_client import call_llm_stream

        mock_litellm.acompletion = AsyncMock(side_effect=Exception("Stream error"))
        with pytest.raises(HTTPException) as exc:
            await call_llm_stream([{"role": "user", "content": "Hi"}])
        assert exc.value.status_code == 500


class TestRetryLogic:
    @patch("app.services.llm_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_rate_limit_then_succeeds(self, mock_sleep):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        mock_message = MagicMock()
        mock_message.content = "retried response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        success_response = MagicMock(choices=[mock_choice])

        orig = llm_module.acompletion
        llm_module.acompletion = AsyncMock(
            side_effect=[
                litellm.RateLimitError("rate limited", "openai", "gpt-4o", response=MagicMock(status_code=429)),
                litellm.RateLimitError("rate limited", "openai", "gpt-4o", response=MagicMock(status_code=429)),
                success_response,
            ]
        )

        try:
            result = await call_llm([{"role": "user", "content": "Hi"}])
            assert result.content == "retried response"
            assert llm_module.acompletion.call_count == 3
            mock_sleep.assert_awaited()
        finally:
            llm_module.acompletion = orig

    @patch("app.services.llm_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_max_retries_exhausted_returns_503(self, mock_sleep):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        orig = llm_module.acompletion
        llm_module.acompletion = AsyncMock(
            side_effect=litellm.RateLimitError("always limited", "openai", "gpt-4o", response=MagicMock(status_code=429))
        )

        try:
            with pytest.raises(HTTPException) as exc:
                await call_llm([{"role": "user", "content": "Hi"}])
            assert exc.value.status_code == 503
            assert "unavailable" in exc.value.detail.lower()
        finally:
            llm_module.acompletion = orig

    async def test_non_retryable_error_raises_immediately(self):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        orig = llm_module.acompletion
        llm_module.acompletion = AsyncMock(side_effect=ValueError("bad input"))

        try:
            with pytest.raises(HTTPException) as exc:
                await call_llm([{"role": "user", "content": "Hi"}])
            assert exc.value.status_code == 500
        finally:
            llm_module.acompletion = orig


class TestFallbackChain:
    @patch("app.services.llm_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_fallback_on_rate_limit(self, mock_sleep):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        mock_message = MagicMock()
        mock_message.content = "fallback response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        success_response = MagicMock(choices=[mock_choice])

        orig = llm_module.acompletion
        # Primary "claude-3-5-sonnet" fails with rate limit x3, fallback "claude-3-7-sonnet" succeeds
        llm_module.acompletion = AsyncMock(
            side_effect=[
                litellm.RateLimitError("rate limited", "openai", "claude-3-5-sonnet", response=MagicMock(status_code=429)),
                litellm.RateLimitError("rate limited", "openai", "claude-3-5-sonnet", response=MagicMock(status_code=429)),
                litellm.RateLimitError("rate limited", "openai", "claude-3-5-sonnet", response=MagicMock(status_code=429)),
                success_response,
            ]
        )

        try:
            result = await call_llm([{"role": "user", "content": "Hi"}], model="claude-3-5-sonnet")
            assert result.content == "fallback response"
            assert llm_module.acompletion.call_count == 4  # 3 retries on primary + 1 on fallback
        finally:
            llm_module.acompletion = orig

    @patch("app.services.llm_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_fallbacks_exhausted_returns_503(self, mock_sleep):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        orig = llm_module.acompletion
        llm_module.acompletion = AsyncMock(
            side_effect=litellm.RateLimitError("always limited", "openai", "gpt-4o", response=MagicMock(status_code=429))
        )

        try:
            with pytest.raises(HTTPException) as exc:
                await call_llm([{"role": "user", "content": "Hi"}], model="gpt-4o")
            assert exc.value.status_code == 503
            # gpt-4o tries 3 times, then gpt-4o-mini tries 3 times, then ollama/llama3.2 tries 3 times = 9 total
            assert llm_module.acompletion.call_count == 9
        finally:
            llm_module.acompletion = orig

    async def test_no_fallback_for_unknown_model(self):
        from app.services.llm_client import call_llm
        from app.services.llm_client import litellm as llm_module

        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message

        orig = llm_module.acompletion
        llm_module.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

        try:
            result = await call_llm([{"role": "user", "content": "Hi"}], model="unknown-model")
            assert result.content == "ok"
            assert llm_module.acompletion.call_count == 1
        finally:
            llm_module.acompletion = orig


class TestTools:
    def test_tools_defined_correctly(self):
        from app.services.llm_client import tools
        assert len(tools) == 2
        names = [t["function"]["name"] for t in tools]
        assert "web_search" in names
        assert "read_knowledge_base" in names
