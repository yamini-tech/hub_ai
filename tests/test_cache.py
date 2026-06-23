import time
from unittest.mock import MagicMock


class TestLlmCache:
    def test_cache_hit_returns_stored(self):
        from app.services.cache import get_cache, set_cache, clear_cache
        clear_cache()
        msg = MagicMock()
        msg.content = "cached result"
        messages = [{"role": "user", "content": "hello"}]
        set_cache(messages, "test-model", msg)
        cached = get_cache(messages, "test-model")
        assert cached is msg

    def test_cache_miss_returns_none(self):
        from app.services.cache import get_cache, clear_cache
        clear_cache()
        result = get_cache([{"role": "user", "content": "no-cache"}], "any-model")
        assert result is None

    def test_cache_key_differentiates_models(self):
        from app.services.cache import get_cache, set_cache, clear_cache
        clear_cache()
        msg = MagicMock()
        msg.content = "model-a result"
        messages = [{"role": "user", "content": "hi"}]
        set_cache(messages, "model-a", msg)
        cached_b = get_cache(messages, "model-b")
        assert cached_b is None

    def test_cache_ttl_expires(self, monkeypatch):
        from app.services.cache import get_cache, set_cache, clear_cache
        clear_cache()
        msg = MagicMock()
        msg.content = "stale"
        messages = [{"role": "user", "content": "ttl-test"}]
        monkeypatch.setattr("app.services.cache._CACHE_TTL_SEC", -1)
        set_cache(messages, "ttl-model", msg)
        cached = get_cache(messages, "ttl-model")
        assert cached is None

    def test_cache_stats(self):
        from app.services.cache import get_cache, set_cache, clear_cache, get_cache_stats
        clear_cache()
        stats = get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        msg = MagicMock()
        msg.content = "x"
        set_cache([{"role": "user", "content": "stats"}], "stats-model", msg)
        stats = get_cache_stats()
        assert stats["size"] == 1

        get_cache([{"role": "user", "content": "stats"}], "stats-model")
        stats = get_cache_stats()
        assert stats["hits"] == 1

        get_cache([{"role": "user", "content": "miss-stats"}], "other-model")
        stats = get_cache_stats()
        assert stats["misses"] >= 1
