import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.schemas import TaskType


class TestGateway:
    def test_gateway_summarize_stream(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "summarize", "text": "Summarize this."},
        )
        assert response.status_code == 200

    def test_gateway_summarize_sync(self, client, mock_litellm_acompletion, mock_tiktoken):
        mock_msg = MagicMock()
        mock_msg.content = "Summary result."
        mock_litellm_acompletion.return_value.choices[0].message = mock_msg

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "summarize", "text": "Summarize this.", "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data

    def test_gateway_parse_stream(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "parse", "text": "Parse this."},
        )
        assert response.status_code == 200

    def test_gateway_parse_sync(self, client, mock_litellm_acompletion, mock_tiktoken):
        mock_msg = MagicMock()
        mock_msg.content = '{"parsed": "data"}'
        mock_litellm_acompletion.return_value.choices[0].message = mock_msg

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "parse", "text": "Parse this.", "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "parsed" in data

    def test_gateway_agent_stream(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "agent", "text": "Hello"},
        )
        assert response.status_code == 200

    def test_gateway_agent_sync(self, client, mock_litellm_acompletion, mock_tiktoken):
        mock_msg = MagicMock()
        mock_msg.content = "Agent response."
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_litellm_acompletion.return_value = MagicMock(choices=[mock_choice])

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "agent", "text": "Hello", "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_gateway_process(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "process", "text": "Process this."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"

    def test_gateway_unknown_task_type(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "unknown", "text": "test"},
        )
        assert response.status_code == 422

    def test_gateway_token_limit_exceeded(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/gateway",
            json={"task_type": "summarize", "text": "x" * 500},
        )
        assert response.status_code == 429
