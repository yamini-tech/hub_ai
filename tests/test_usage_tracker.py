from app.services.usage_tracker import (
    get_api_key,
    get_request_id,
    pop_usage,
    record_llm_usage,
    set_api_key,
    set_request_id,
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
