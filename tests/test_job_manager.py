import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from app.services.job_manager import (
    create_job, update_job, get_job,
    add_message_to_history, get_history, _memory_store,
)


def _reset_store():
    _memory_store.clear()


class TestJobManager:
    def test_create_job_returns_uuid(self):
        _reset_store()
        job_id = create_job()
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_create_job_stores_pending(self):
        _reset_store()
        job_id = create_job()
        result = get_job(job_id)
        assert result is not None
        assert result["status"] == "pending"
        assert result["result"] is None

    def test_update_job_and_get_job(self):
        _reset_store()
        job_id = create_job()
        update_job(job_id, "completed", {"response": "done"})
        result = get_job(job_id)
        assert result["status"] == "completed"
        assert result["result"]["response"] == "done"

    def test_get_job_not_found(self):
        _reset_store()
        result = get_job("nonexistent-id")
        assert result is None

    @patch("app.services.job_manager._get_redis")
    def test_create_job_uses_redis_when_available(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        job_id = create_job()
        mock_redis.set.assert_called_once()
        args, _ = mock_redis.set.call_args
        assert args[0] == f"job:{job_id}"
        assert json.loads(args[1])["status"] == "pending"

    @patch("app.services.job_manager._get_redis")
    def test_get_job_uses_redis_when_available(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"status": "done", "result": "ok"})
        mock_get_redis.return_value = mock_redis
        result = get_job("test-id")
        mock_redis.get.assert_called_with("job:test-id")
        assert result["status"] == "done"


class TestChatHistory:
    def test_add_and_get_history(self):
        _reset_store()
        sid = str(uuid.uuid4())
        add_message_to_history(sid, "user", "Hello")
        history = get_history(sid)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_history_limited_to_10(self):
        _reset_store()
        sid = str(uuid.uuid4())
        for i in range(15):
            add_message_to_history(sid, "user", f"msg{i}")
        history = get_history(sid)
        assert len(history) == 10

    def test_get_empty_history(self):
        _reset_store()
        history = get_history(str(uuid.uuid4()))
        assert history == []

    def test_multiple_sessions(self):
        _reset_store()
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        add_message_to_history(sid_a, "user", "A1")
        add_message_to_history(sid_b, "user", "B1")
        add_message_to_history(sid_a, "assistant", "A2")

        hist_a = get_history(sid_a)
        assert len(hist_a) == 2

        hist_b = get_history(sid_b)
        assert len(hist_b) == 1
