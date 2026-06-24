import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
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

    @patch("app.core.security.API_KEY_ENABLED", True)
    @patch("app.core.security.API_KEY", "valid-key-123")
    @patch("app.core.security.RATE_LIMIT_MAX_REQUESTS", 5)
    @patch("app.services.throttling._get_rate_limiter", return_value=None)
    async def test_rate_limit_exceeded_raises_429(self, mock_limiter):
        from app.core.security import verify_api_key
        from app.services.throttling import _rate_windows
        _rate_windows["apikey:valid-ke"] = [time.time() - 1 for _ in range(5)]
        mock_request = MagicMock()
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(mock_request, api_key="valid-key-123")
        assert exc.value.status_code == 429

    @patch("app.core.security.API_KEY_ENABLED", True)
    @patch("app.core.security.API_KEY", "valid-key-123")
    @patch("app.core.security.RATE_LIMIT_MAX_REQUESTS", 5)
    @patch("app.services.throttling._get_rate_limiter", return_value=None)
    async def test_within_key_rate_limit_passes(self, mock_limiter):
        from app.core.security import verify_api_key
        from app.services.throttling import _rate_windows
        _rate_windows["apikey:valid-ke"] = [time.time() - 1 for _ in range(3)]
        mock_request = MagicMock()
        result = await verify_api_key(mock_request, api_key="valid-key-123")
        assert result is True
