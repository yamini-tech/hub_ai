from app.main import app


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


class TestGlobalExceptionHandler:
    def test_global_handler_registered(self, client):
        from app.main import app as _app
        handlers = _app.exception_handlers
        assert Exception in handlers
        assert callable(handlers[Exception])
