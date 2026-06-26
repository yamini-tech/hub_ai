import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException


class TestVerifyApiKey:
    @patch("app.core.security.API_KEY_ENABLED", False)
    async def test_disabled_returns_true(self):
        from app.core.security import verify_api_key
        mock_request = MagicMock()
        result = await verify_api_key(mock_request, api_key="")
        assert result is True

    @patch("app.core.security.API_KEY_ENABLED", True)
    @patch("app.core.security.API_KEY", "valid-key-123")
    async def test_valid_key_returns_true(self):
        from app.core.security import verify_api_key
        mock_request = MagicMock()
        result = await verify_api_key(mock_request, api_key="valid-key-123")
        assert result is True

    @patch("app.core.security.API_KEY_ENABLED", True)
    @patch("app.core.security.API_KEY", "valid-key-123")
    async def test_missing_key_raises_403(self):
        from app.core.security import verify_api_key
        mock_request = MagicMock()
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(mock_request, api_key=None)
        assert exc.value.status_code == 403

    @patch("app.core.security.API_KEY_ENABLED", True)
    @patch("app.core.security.API_KEY", "valid-key-123")
    async def test_wrong_key_raises_403(self):
        from app.core.security import verify_api_key
        mock_request = MagicMock()
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(mock_request, api_key="wrong-key")
        assert exc.value.status_code == 403


