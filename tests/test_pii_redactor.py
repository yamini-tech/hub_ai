import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services.pii_redactor import redact_pii, redact_stream


class TestPiiRedactor:
    def test_redacts_email(self):
        result = redact_pii("contact me at user@example.com please")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_redacts_phone(self):
        result = redact_pii("call +1-555-123-4567 now")
        assert "[PHONE]" in result

    def test_redacts_ssn(self):
        result = redact_pii("SSN: 123-45-6789")
        assert "[SSN]" in result

    def test_redacts_credit_card(self):
        result = redact_pii("card 4111-1111-1111-1111")
        assert "[CC]" in result

    def test_no_pii_unchanged(self):
        result = redact_pii("hello world this is safe")
        assert result == "hello world this is safe"

    def test_non_string_returns_as_is(self):
        assert redact_pii(123) == 123
        assert redact_pii(None) is None


class TestRedactStream:
    @pytest.mark.asyncio
    async def test_redacts_email_in_stream(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "email is user@example.com"
            yield chunk

        results = []
        async for c in redact_stream(_mock_stream()):
            results.append(c.choices[0].delta.content)
        assert "[EMAIL]" in results[0]
        assert "user@example.com" not in results[0]

    @pytest.mark.asyncio
    async def test_redacts_phone_in_stream(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "call +1-555-123-4567"
            yield chunk

        async for c in redact_stream(_mock_stream()):
            assert "[PHONE]" in c.choices[0].delta.content

    @pytest.mark.asyncio
    async def test_safe_content_unchanged(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "hello world"
            yield chunk

        async for c in redact_stream(_mock_stream()):
            assert c.choices[0].delta.content == "hello world"

    @pytest.mark.asyncio
    async def test_handles_empty_delta(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            yield chunk

        async for c in redact_stream(_mock_stream()):
            assert c.choices[0].delta.content is None

    @pytest.mark.asyncio
    async def test_handles_no_choices(self):
        async def _mock_stream():
            chunk = MagicMock()
            chunk.choices = []
            yield chunk

        results = []
        async for c in redact_stream(_mock_stream()):
            results.append(c)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_call_llm_redacts_pii(self):
        from unittest.mock import patch
        from app.services.llm_client import call_llm

        with patch("app.services.llm_client.litellm") as mock_litellm:
            mock_message = MagicMock()
            mock_message.content = "email is user@example.com"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_litellm.acompletion = AsyncMock(return_value=MagicMock(choices=[mock_choice]))

            result = await call_llm([{"role": "user", "content": "Hi"}])
            assert "[EMAIL]" in result.content
            assert "user@example.com" not in result.content
