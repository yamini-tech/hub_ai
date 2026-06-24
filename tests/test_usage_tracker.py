from app.services.usage_tracker import (
    set_request_id, get_request_id,
    set_api_key, get_api_key,
    record_llm_usage, pop_usage, get_user_usage,
)


class TestUsageTracker:
    def setup_method(self):
        set_request_id("")
        set_api_key(None)

    def test_set_and_get_request_id(self):
        set_request_id("req-123")
        assert get_request_id() == "req-123"

    def test_get_request_id_no_context(self):
        from app.services.usage_tracker import _usage_store
        _usage_store.clear()
        # When setup_method set it to "", get_request_id returns ""
        assert get_request_id() == ""

    def test_set_and_get_api_key(self):
        set_api_key("sk-test-key")
        assert get_api_key() == "sk-test-key"

    def test_record_and_pop_usage(self):
        from app.services.usage_tracker import _usage_store
        _usage_store.clear()
        set_request_id("req-1")
        record_llm_usage(model="gpt-4o", prompt_tokens=10, completion_tokens=20)
        usage = pop_usage("req-1")
        assert usage is not None
        assert usage["model"] == "gpt-4o"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_record_no_request_id(self):
        from app.services.usage_tracker import _usage_store
        _usage_store.clear()
        set_request_id("")
        # Should not raise
        record_llm_usage(model="gpt-4o", prompt_tokens=10, completion_tokens=20)

    def test_pop_nonexistent(self):
        assert pop_usage("nonexistent") is None

    def test_get_user_usage_no_key(self):
        assert get_user_usage(None) is None
        assert get_user_usage("") is None

    def test_get_user_usage_records(self):
        from app.services.usage_tracker import _user_usage_store
        _user_usage_store.clear()
        set_request_id("req-2")
        set_api_key("sk-test-api-key-12345")
        record_llm_usage(model="gpt-4o", prompt_tokens=10, completion_tokens=20)
        result = get_user_usage("sk-test-api-key-12345")
        assert result is not None
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20

    def test_user_usage_aggregates(self):
        from app.services.usage_tracker import _usage_store, _user_usage_store
        _usage_store.clear()
        _user_usage_store.clear()
        set_api_key("sk-test-key")
        set_request_id("req-3")
        record_llm_usage(model="gpt-4o", prompt_tokens=5, completion_tokens=5)
        set_request_id("req-4")
        record_llm_usage(model="gpt-4o", prompt_tokens=5, completion_tokens=5)
        result = get_user_usage("sk-test-key")
        assert result is not None
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 10
        assert result["total_tokens"] == 20
