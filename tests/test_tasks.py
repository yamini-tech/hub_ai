import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestSummarizeEndpoint:
    def test_summarize_stream_returns_sse(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/summarize",
            json={"text": "Summarize this text for me."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_summarize_sync_returns_json(self, client, mock_litellm_acompletion, mock_tiktoken):
        mock_msg = MagicMock()
        mock_msg.content = "This is a summary."
        mock_litellm_acompletion.return_value.choices[0].message = mock_msg

        response = client.post(
            "/api/ai/summarize/sync",
            json={"text": "Summarize this text for me."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data

    def test_summarize_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 25000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/summarize",
            json={"text": "x" * 1000},
        )
        assert response.status_code == 429


class TestParseEndpoint:
    def test_parse_stream_returns_sse(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/parse",
            json={"text": "Parse this unstructured text."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_parse_sync_returns_json(self, client, mock_litellm_acompletion, mock_tiktoken):
        mock_msg = MagicMock()
        mock_msg.content = '{"key": "value"}'
        mock_litellm_acompletion.return_value.choices[0].message = mock_msg

        response = client.post(
            "/api/ai/parse/sync",
            json={"text": "Parse this text."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "parsed" in data

    def test_parse_with_schema_hint(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/parse/sync",
            json={"text": "Parse this.", "schema_hint": "JSON"},
        )
        assert response.status_code == 200

    def test_parse_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 25000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/parse",
            json={"text": "x" * 1000},
        )
        assert response.status_code == 429

    def test_summarize_sync_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 25000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/summarize/sync",
            json={"text": "x" * 1000},
        )
        assert response.status_code == 429

    def test_parse_sync_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 25000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/parse/sync",
            json={"text": "x" * 1000},
        )
        assert response.status_code == 429
