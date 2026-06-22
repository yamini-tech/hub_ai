import json
import uuid
import os
import threading
from typing import Any

_REDIS_AVAILABLE = False
_redis = None

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
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

# In-memory fallback
_memory_store: dict[str, Any] = {}
_memory_lock = threading.Lock()

def create_job() -> str:
    job_id = str(uuid.uuid4())
    data = json.dumps({"status": "pending", "result": None})
    if _REDIS_AVAILABLE:
        _redis.set(f"job:{job_id}", data, ex=3600)
    else:
        with _memory_lock:
            _memory_store[f"job:{job_id}"] = data
    return job_id

def update_job(job_id: str, status: str, result: Any = None):
    data = json.dumps({"status": status, "result": result})
    if _REDIS_AVAILABLE:
        _redis.set(f"job:{job_id}", data, ex=3600)
    else:
        with _memory_lock:
            _memory_store[f"job:{job_id}"] = data

def get_job(job_id: str) -> dict | None:
    if _REDIS_AVAILABLE:
        data = _redis.get(f"job:{job_id}")
    else:
        with _memory_lock:
            data = _memory_store.get(f"job:{job_id}")
    return json.loads(data) if data else None

def add_message_to_history(session_id: str, role: str, content: str):
    key = f"history:{session_id}"
    message = json.dumps({"role": role, "content": content})
    if _REDIS_AVAILABLE:
        _redis.lpush(key, message)
        _redis.ltrim(key, 0, 9)
    else:
        with _memory_lock:
            history = json.loads(_memory_store.get(key, "[]"))
            history.insert(0, json.loads(message))
            _memory_store[key] = json.dumps(history[:10])

def get_history(session_id: str) -> list:
    key = f"history:{session_id}"
    if _REDIS_AVAILABLE:
        data = _redis.lrange(key, 0, 9)
        return [json.loads(msg) for msg in reversed(data)]
    else:
        with _memory_lock:
            data = _memory_store.get(key, "[]")
            return json.loads(data)
