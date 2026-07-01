from unittest.mock import MagicMock


class TestAgentEndpoint:
    async def _mock_stream(self, messages, tools_list=None, model=None):
        async def generate():
            delta = MagicMock()
            delta.content = "test chunk"
            delta.tool_calls = None
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = delta
            yield chunk

        return generate()

    def test_agent_stream_cross_session_stores_exchange(self, client, mock_tiktoken):
        from unittest.mock import patch

        with patch("app.routers.chat.call_llm_stream", new=self._mock_stream):
            with patch("app.routers.chat.save_to_knowledge_base") as mock_save:
                response = client.post(
                    "/api/agent",
                    json={"text": "Hello", "session_id": "test_sess", "cross_session": True},
                )
                assert response.status_code == 200
                body = b"".join(response.iter_bytes())
                assert b"[DONE]" in body
                mock_save.assert_called_once()
                call_arg = mock_save.call_args[0][0]
                assert "[session:test_sess]" in call_arg
                assert "User: Hello" in call_arg
                assert "Assistant:" in call_arg

    def test_agent_stream_cross_session_false_does_not_store(self, client, mock_tiktoken):
        from unittest.mock import patch

        with patch("app.routers.chat.call_llm_stream", new=self._mock_stream):
            with patch("app.routers.chat.save_to_knowledge_base") as mock_save:
                response = client.post(
                    "/api/agent",
                    json={"text": "Hello", "session_id": "test_sess"},
                )
                assert response.status_code == 200
                body = b"".join(response.iter_bytes())
                assert b"[DONE]" in body
                mock_save.assert_not_called()

    def test_agent_remember_command(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        response = client.post(
            "/api/agent",
            json={"text": "remember this fact", "session_id": "test_sess"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "stored that in my memory" in data["answer"]

    def test_agent_stream_returns_sse(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/agent",
            json={"text": "Hello agent", "session_id": "test_sess"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_agent_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/agent",
            json={"text": "x" * 500, "session_id": "test_sess"},
        )
        assert response.status_code == 429


class TestAgentSyncEndpoint:
    def test_agent_sync_handles_malformed_tool_args(self, client, mock_tiktoken):
        from unittest.mock import MagicMock, patch

        with patch("app.routers.chat.call_llm") as mock_call_llm:
            mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5
            mock_message = MagicMock()
            mock_message.content = None
            mock_message.tool_calls = [
                MagicMock(id="call_1", function=MagicMock(name="web_search", arguments="not valid json{{{"))
            ]
            mock_call_llm.return_value = mock_message

            response = client.post(
                "/api/agent/sync",
                json={"text": "Hello", "session_id": "test_sess"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data

    def test_agent_sync_remember(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        response = client.post(
            "/api/agent/sync",
            json={"text": "remember this fact", "session_id": "test_sess"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "stored that in my memory" in data["answer"]

    def test_agent_sync_returns_answer(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/agent/sync",
            json={"text": "Hello", "session_id": "test_sess"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_agent_sync_cross_session_stores_exchange(self, client, mock_litellm_acompletion, mock_tiktoken):
        from unittest.mock import patch

        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5
        with patch("app.routers.chat.save_to_knowledge_base") as mock_save:
            response = client.post(
                "/api/agent/sync",
                json={"text": "Hello", "session_id": "test_sess", "cross_session": True},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            mock_save.assert_called_once()
            call_arg = mock_save.call_args[0][0]
            assert "[session:test_sess]" in call_arg
            assert "User: Hello" in call_arg
            assert "test response" in call_arg

    def test_agent_sync_cross_session_false_does_not_store(self, client, mock_litellm_acompletion, mock_tiktoken):
        from unittest.mock import patch

        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5
        with patch("app.routers.chat.save_to_knowledge_base") as mock_save:
            response = client.post(
                "/api/agent/sync",
                json={"text": "Hello", "session_id": "test_sess"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            mock_save.assert_not_called()

    def test_agent_sync_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/agent/sync",
            json={"text": "x" * 500, "session_id": "test_sess"},
        )
        assert response.status_code == 429


class TestProcessEndpoint:
    def test_process_creates_job(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        response = client.post(
            "/api/ai/process",
            json={"text": "Process this", "task_type": "summarize"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"

    def test_process_exceeds_token_limit(self, client, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        response = client.post(
            "/api/ai/process",
            json={"text": "x" * 500, "task_type": "summarize"},
        )
        assert response.status_code == 429


class TestStatusEndpoint:
    def test_get_existing_job(self, client, mock_tiktoken, mock_litellm_acompletion):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5

        create_resp = client.post(
            "/api/ai/process",
            json={"text": "Process this", "task_type": "summarize"},
        )
        job_id = create_resp.json()["job_id"]

        import time

        time.sleep(0.5)

        response = client.get(f"/api/ai/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/ai/status/nonexistent-id")
        assert response.status_code == 404
