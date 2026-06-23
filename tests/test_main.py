from app.main import app


class TestGracefulShutdown:
    def test_shutdown_handler_registered(self):
        from app.main import app as _app
        assert len(_app.router.on_shutdown) == 1


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SmartHub AI Brain"
        assert data["version"] == "2.0.0"
        assert data["status"] == "online"
        endpoints = data.get("endpoints", {})
        assert "health" in endpoints
        assert "ready" in endpoints

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "SmartHub AI"

    def test_docs_redirect(self, client):
        response = client.get("/docs")
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestReadyEndpoint:
    def test_ready_returns_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["prompts_loaded"] is True


class TestRequestBodySizeLimit:
    def test_small_request_passes(self, client, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5
        response = client.post("/api/ai/process", json={"text": "small", "task_type": "summarize"})
        assert response.status_code == 202

    def test_oversized_request_returns_413(self, client, monkeypatch, mock_tiktoken):
        mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 5
        monkeypatch.setattr("app.main.MAX_REQUEST_SIZE", 100)
        response = client.post(
            "/api/ai/process",
            json={"text": "x" * 200, "task_type": "summarize"},
        )
        assert response.status_code == 413
        data = response.json()
        assert "detail" in data


class TestGlobalExceptionHandler:
    def test_global_handler_registered(self, client):
        from app.main import app as _app
        handlers = _app.exception_handlers
        assert Exception in handlers
        assert callable(handlers[Exception])
