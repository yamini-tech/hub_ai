import time
import os
import tiktoken
from collections import defaultdict

# Per-session rate tracking (in-memory fallback)
_rate_windows: dict[str, list[float]] = defaultdict(list)

# Optional Redis-backed rate limiting for multi-worker support
_REDIS_RATE_LIMITER = None


def _get_rate_limiter():
    global _REDIS_RATE_LIMITER
    if _REDIS_RATE_LIMITER is None:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        try:
            import redis as redis_module
            r = redis_module.Redis(
                host=redis_host, port=redis_port, db=0,
                decode_responses=True, socket_connect_timeout=1,
            )
            r.ping()
            _REDIS_RATE_LIMITER = r
        except Exception:
            _REDIS_RATE_LIMITER = False
    return _REDIS_RATE_LIMITER if _REDIS_RATE_LIMITER else None


def get_token_count(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def is_request_allowed(text: str, max_tokens: int = 10000) -> tuple[bool, int]:
    count = get_token_count(text)
    return count <= max_tokens, count

def check_rate_limit(session_id: str, window_sec: int = 60, max_requests: int = 30) -> tuple[bool, int]:
    r = _get_rate_limiter()
    if r is not None:
        return _check_rate_limit_redis(r, session_id, window_sec, max_requests)
    return _check_rate_limit_memory(session_id, window_sec, max_requests)


def check_rate_limit_by_key(api_key: str, window_sec: int = 60, max_requests: int = 30) -> tuple[bool, int]:
    if not api_key:
        return True, 0
    key = f"apikey:{api_key[:8]}"
    r = _get_rate_limiter()
    if r is not None:
        return _check_rate_limit_redis(r, key, window_sec, max_requests)
    return _check_rate_limit_memory(key, window_sec, max_requests)


def _check_rate_limit_redis(r, session_id: str, window_sec: int, max_requests: int) -> tuple[bool, int]:
    key = f"rate:{session_id}"
    now = int(time.time())
    window_start = now - window_sec
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zcard(key)
    pipe.zadd(key, {str(now): now})
    pipe.expire(key, window_sec)
    results = pipe.execute()
    count = int(results[1]) + 1
    if count > max_requests:
        return False, count
    return True, count


def _check_rate_limit_memory(session_id: str, window_sec: int, max_requests: int) -> tuple[bool, int]:
    now = time.time()
    window_start = now - window_sec
    timestamps = _rate_windows[session_id]
    _rate_windows[session_id] = [t for t in timestamps if t > window_start]
    count = len(_rate_windows[session_id])
    if count >= max_requests:
        return False, count
    _rate_windows[session_id].append(now)
    return True, count + 1
