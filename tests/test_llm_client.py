import pytest
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
    async def test_call_llm_stream_returns_async_iterable(self, mock_litellm):
        from app.services.llm_client import call_llm_stream

        mock_stream = AsyncMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_stream)

        result = await call_llm_stream([{"role": "user", "content": "Hi"}])
        assert result == mock_stream
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


class TestTools:
    def test_tools_defined_correctly(self):
        from app.services.llm_client import tools
        assert len(tools) == 2
        names = [t["function"]["name"] for t in tools]
        assert "web_search" in names
        assert "read_knowledge_base" in names
