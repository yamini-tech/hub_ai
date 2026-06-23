import json
import uuid
import os
import threading
import time
from typing import Any

_redis = None
_redis_lock = threading.Lock()
_redis_last_attempt = 0.0
_REDIS_RETRY_INTERVAL = 30

# In-memory fallback
_memory_store: dict[str, Any] = {}
_memory_lock = threading.Lock()


def _get_redis():
    global _redis, _redis_last_attempt
    if _redis is not None:
        try:
            _redis.ping()
            return _redis
        except Exception:
            _redis = None
    now = time.time()
    if now - _redis_last_attempt < _REDIS_RETRY_INTERVAL:
        return None
    with _redis_lock:
        if _redis is not None:
            return _redis
        if now - _redis_last_attempt < _REDIS_RETRY_INTERVAL:
            return None
        _redis_last_attempt = now
        try:
            import redis as redis_module
            _redis = redis_module.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=0,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            _redis.ping()
        except Exception:
            _redis = None
    return _redis

def _redis_set(key: str, value: str, ex: int = 3600) -> bool:
    r = _get_redis()
    if r is not None:
        try:
            r.set(key, value, ex=ex)
            return True
        except Exception:
            pass
    return False


def _redis_get(key: str) -> str | None:
    r = _get_redis()
    if r is not None:
        try:
            return r.get(key)
        except Exception:
            pass
    return None


def _redis_lpush_trim(key: str, value: str, max_len: int = 10) -> bool:
    r = _get_redis()
    if r is not None:
        try:
            r.lpush(key, value)
            r.ltrim(key, 0, max_len - 1)
            return True
        except Exception:
            pass
    return False


def _redis_lrange(key: str, start: int = 0, end: int = 9) -> list[str] | None:
    r = _get_redis()
    if r is not None:
        try:
            return r.lrange(key, start, end)
        except Exception:
            pass
    return None


def create_job() -> str:
    job_id = str(uuid.uuid4())
    data = json.dumps({"status": "pending", "result": None})
    if not _redis_set(f"job:{job_id}", data):
        with _memory_lock:
            _memory_store[f"job:{job_id}"] = data
    return job_id

def update_job(job_id: str, status: str, result: Any = None):
    data = json.dumps({"status": status, "result": result})
    if not _redis_set(f"job:{job_id}", data):
        with _memory_lock:
            _memory_store[f"job:{job_id}"] = data

def get_job(job_id: str) -> dict | None:
    data = _redis_get(f"job:{job_id}")
    if data is None:
        with _memory_lock:
            data = _memory_store.get(f"job:{job_id}")
    return json.loads(data) if data else None

def add_message_to_history(session_id: str, role: str, content: str):
    key = f"history:{session_id}"
    message = json.dumps({"role": role, "content": content})
    if not _redis_lpush_trim(key, message):
        with _memory_lock:
            history = json.loads(_memory_store.get(key, "[]"))
            history.insert(0, json.loads(message))
            _memory_store[key] = json.dumps(history[:10])

def get_history(session_id: str) -> list:
    data = _redis_lrange(key := f"history:{session_id}")
    if data is not None:
        return [json.loads(msg) for msg in reversed(data)]
    with _memory_lock:
        data = _memory_store.get(key, "[]")
        return json.loads(data)
