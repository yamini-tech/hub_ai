import hashlib
import json
import os
import threading
import time

_CACHE_TTL_SEC = int(os.getenv("SMARTHUB_CACHE_TTL", "300"))
_CACHE_MAX_SIZE = 500
_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0}


def _make_key(messages: list, model: str, tools_list=None, response_format=None) -> str:
    raw = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools_list,
            "response_format": (
                response_format.model_dump() if hasattr(response_format, "model_dump") else response_format
            ),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cache(messages: list, model: str, tools_list=None, response_format=None):
    key = _make_key(messages, model, tools_list, response_format)
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < _CACHE_TTL_SEC:
            _cache_stats["hits"] += 1
            return entry["response"]
        if entry:
            del _cache[key]
        _cache_stats["misses"] += 1
    return None


def set_cache(messages: list, model: str, response, tools_list=None, response_format=None):
    key = _make_key(messages, model, tools_list, response_format)
    with _cache_lock:
        _cache[key] = {"ts": time.time(), "response": response}
        if len(_cache) > _CACHE_MAX_SIZE:
            oldest = min(_cache.keys(), key=lambda k: _cache[k]["ts"])
            del _cache[oldest]


def clear_cache():
    with _cache_lock:
        _cache.clear()
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0


def get_cache_stats():
    with _cache_lock:
        return dict(_cache_stats, size=len(_cache))
