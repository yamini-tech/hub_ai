from unittest.mock import MagicMock


class TestLlmCache:
    def test_cache_hit_returns_stored(self):
        from app.services.cache import clear_cache, get_cache, set_cache

        clear_cache()
        msg = MagicMock()
        msg.content = "cached result"
        messages = [{"role": "user", "content": "hello"}]
        set_cache(messages, "test-model", msg)
        cached = get_cache(messages, "test-model")
        assert cached is msg

    def test_cache_miss_returns_none(self):
        from app.services.cache import clear_cache, get_cache

        clear_cache()
        result = get_cache([{"role": "user", "content": "no-cache"}], "any-model")
        assert result is None

    def test_cache_key_differentiates_models(self):
        from app.services.cache import clear_cache, get_cache, set_cache

        clear_cache()
        msg = MagicMock()
        msg.content = "model-a result"
        messages = [{"role": "user", "content": "hi"}]
        set_cache(messages, "model-a", msg)
        cached_b = get_cache(messages, "model-b")
        assert cached_b is None

    def test_cache_ttl_expires(self, monkeypatch):
        from app.services.cache import clear_cache, get_cache, set_cache

        clear_cache()
        msg = MagicMock()
        msg.content = "stale"
        messages = [{"role": "user", "content": "ttl-test"}]
        monkeypatch.setattr("app.services.cache._CACHE_TTL_SEC", -1)
        set_cache(messages, "ttl-model", msg)
        cached = get_cache(messages, "ttl-model")
        assert cached is None
