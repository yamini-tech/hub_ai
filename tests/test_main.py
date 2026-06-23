from app.main import app


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SmartHub AI Brain"
        assert data["version"] == "2.0.0"
        assert data["status"] == "online"
        assert "endpoints" in data

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "SmartHub AI"

    def test_docs_redirect(self, client):
        response = client.get("/docs")
        assert response.status_code == 200


class TestGlobalExceptionHandler:
    def test_global_handler_registered(self, client):
        from app.main import app as _app
        handlers = _app.exception_handlers
        assert Exception in handlers
        assert callable(handlers[Exception])
