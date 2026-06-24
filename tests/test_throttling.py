import time
import pytest
from unittest.mock import patch, MagicMock
import app.services.throttling as throttling_mod
from app.services.throttling import get_token_count, is_request_allowed, check_rate_limit, check_rate_limit_by_key, _rate_windows


@pytest.fixture(autouse=True)
def force_memory_rate_limiter():
    with patch.object(throttling_mod, "_get_rate_limiter", return_value=None):
        yield


class TestGetTokenCount:
    def test_with_known_model(self, mock_tiktoken):
        count = get_token_count("hello world", model="gpt-4o-mini")
        mock_tiktoken.encoding_for_model.assert_called_with("gpt-4o-mini")
        assert count == 10

    def test_with_unknown_model_fallback(self, mock_tiktoken):
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown")
        mock_fallback = MagicMock()
        mock_fallback.encode.return_value = [1] * 5
        mock_tiktoken.get_encoding.return_value = mock_fallback

        count = get_token_count("test text", model="unknown-model")
        mock_tiktoken.get_encoding.assert_called_with("cl100k_base")
        assert count == 5


class TestIsRequestAllowed:
    def test_within_limit(self, mock_tiktoken):
        allowed, count = is_request_allowed("short text", max_tokens=10000)
        assert allowed is True
        assert count == 10

    def test_exceeds_limit(self, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        allowed, count = is_request_allowed("long text", max_tokens=10000)
        assert allowed is False
        assert count == 15000

    def test_empty_text_allowed(self, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = []
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        allowed, count = is_request_allowed("", max_tokens=10000)
        assert allowed is True
        assert count == 0

    def test_exact_limit_allowed(self, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 10000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        allowed, count = is_request_allowed("x" * 5000, max_tokens=10000)
        assert allowed is True
        assert count == 10000

    def test_one_over_limit_denied(self, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 10001
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        allowed, count = is_request_allowed("x" * 5000, max_tokens=10000)
        assert allowed is False
        assert count == 10001

    def test_different_limits(self, mock_tiktoken):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 15000
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        allowed_10k, _ = is_request_allowed("x" * 5000, max_tokens=10000)
        assert allowed_10k is False

        allowed_20k, _ = is_request_allowed("x" * 5000, max_tokens=20000)
        assert allowed_20k is True


class TestCheckRateLimit:
    def setup_method(self):
        _rate_windows.clear()

    def test_first_request_allowed(self):
        ok, count = check_rate_limit("session1", window_sec=60, max_requests=5)
        assert ok is True
        assert count == 1

    def test_within_limit(self):
        _rate_windows["session1"] = [time.time() - 1 for _ in range(3)]
        ok, count = check_rate_limit("session1", window_sec=60, max_requests=5)
        assert ok is True
        assert count == 4

    def test_exceeds_limit(self):
        _rate_windows["session1"] = [time.time() - 1 for _ in range(5)]
        ok, count = check_rate_limit("session1", window_sec=60, max_requests=5)
        assert ok is False
        assert count == 5

    def test_prunes_old_entries(self):
        old = time.time() - 120
        recent = time.time() - 1
        _rate_windows["session1"] = [old, old, recent]
        ok, count = check_rate_limit("session1", window_sec=60, max_requests=5)
        assert ok is True
        assert count == 2

    def test_multiple_sessions_independent(self):
        _rate_windows["session_a"] = [time.time() - 1 for _ in range(5)]
        ok_a, count_a = check_rate_limit("session_a", window_sec=60, max_requests=5)
        assert ok_a is False
        assert count_a == 5

        ok_b, count_b = check_rate_limit("session_b", window_sec=60, max_requests=5)
        assert ok_b is True
        assert count_b == 1


class TestCheckRateLimitByKey:
    def setup_method(self):
        _rate_windows.clear()

    def test_empty_key_returns_ok(self):
        ok, count = check_rate_limit_by_key("")
        assert ok is True
        assert count == 0

    def test_none_key_returns_ok(self):
        ok, count = check_rate_limit_by_key(None)
        assert ok is True
        assert count == 0

    def test_first_request_allowed(self):
        ok, count = check_rate_limit_by_key("sk-test-key-12345", window_sec=60, max_requests=5)
        assert ok is True
        assert count == 1

    def test_within_limit(self):
        ok, count = check_rate_limit_by_key("sk-test-key-12345", window_sec=60, max_requests=5)
        assert ok is True
        ok, count = check_rate_limit_by_key("sk-test-key-12345", window_sec=60, max_requests=5)
        assert ok is True
        assert count == 2

    def test_exceeds_limit(self):
        _rate_windows["apikey:sk-test-"] = [time.time() - 1 for _ in range(5)]
        ok, count = check_rate_limit_by_key("sk-test-key-12345", window_sec=60, max_requests=5)
        assert ok is False
        assert count == 5
