import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestChatStream:
    def test_chat_stream_success(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/v1/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}], "user_id": "test_user"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_chat_stream_missing_messages(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/chat/stream",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422

    def test_chat_stream_with_rag(self, client, mock_litellm_acompletion, mock_tiktoken):
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "user_id": "test_user",
                "use_rag": True,
            },
        )
        assert response.status_code == 200


class TestEmbed:
    def test_embed_success(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/embed",
            json={"text": "Embed this text", "user_id": "test_user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)

    def test_embed_missing_text(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/embed",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422


class TestExtract:
    def test_extract_missing_params(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/extract",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422

    def test_extract_file_not_found(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/extract",
            json={
                "file_path": "/nonexistent/file.txt",
                "file_type": "txt",
                "user_id": "test_user",
            },
        )
        assert response.status_code == 404

    def test_extract_unsupported_type(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/extract",
            json={
                "file_path": "/fake/file.xyz",
                "file_type": "xyz",
                "user_id": "test_user",
            },
        )
        assert response.status_code == 400

    def test_extract_txt_success(self, client, mock_tiktoken):
        import tempfile, os
        content = "Extracted text content."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            response = client.post(
                "/api/v1/extract",
                json={
                    "file_path": path,
                    "file_type": "txt",
                    "user_id": "test_user",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == content
        finally:
            os.unlink(path)


class TestRagIngest:
    @patch("app.routers.hub_compat.chunk_text")
    @patch("app.routers.hub_compat.ingest_chunks")
    def test_rag_ingest_success(self, mock_ingest, mock_chunk, client, mock_tiktoken):
        mock_chunk.return_value = ["chunk1", "chunk2"]
        mock_ingest.return_value = 2

        response = client.post(
            "/api/v1/rag/ingest",
            json={
                "user_id": "test_user",
                "document_id": "doc_1",
                "text": "Some text to ingest.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_stored"] == 2

    def test_rag_ingest_missing_params(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/rag/ingest",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422


class TestRagRetrieve:
    @patch("app.routers.hub_compat.retrieve_chunks")
    def test_rag_retrieve_success(self, mock_retrieve, client, mock_tiktoken):
        mock_retrieve.return_value = ["chunk1", "chunk2"]

        response = client.post(
            "/api/v1/rag/retrieve",
            json={
                "user_id": "test_user",
                "query": "test query",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert len(data["chunks"]) == 2

    def test_rag_retrieve_missing_params(self, client, mock_tiktoken):
        response = client.post(
            "/api/v1/rag/retrieve",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422


class TestRagDelete:
    @patch("app.routers.hub_compat.delete_document_chunks")
    def test_rag_delete_success(self, mock_delete, client, mock_tiktoken):
        response = client.delete(
            "/api/v1/rag/documents/doc_1?user_id=test_user",
        )
        assert response.status_code == 204

    def test_rag_delete_missing_user_id(self, client, mock_tiktoken):
        response = client.delete("/api/v1/rag/documents/doc_1")
        assert response.status_code in (400, 422)
