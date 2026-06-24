import contextvars
import threading

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")
api_key_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("api_key", default=None)

_usage_store: dict[str, dict] = {}
_user_usage_store: dict[str, dict] = {}
_lock = threading.Lock()


def set_request_id(request_id: str):
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    try:
        return request_id_var.get()
    except LookupError:
        return None


def set_api_key(api_key: str | None):
    api_key_var.set(api_key)


def get_api_key() -> str | None:
    try:
        return api_key_var.get()
    except LookupError:
        return None


def record_llm_usage(model: str, prompt_tokens: int, completion_tokens: int):
    rid = get_request_id()
    if not rid:
        return
    api_key = get_api_key()
    with _lock:
        existing = _usage_store.get(
            rid,
            {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        existing["prompt_tokens"] += prompt_tokens
        existing["completion_tokens"] += completion_tokens
        existing["total_tokens"] += prompt_tokens + completion_tokens
        existing["model"] = model
        existing["api_key"] = api_key
        _usage_store[rid] = existing

        if api_key:
            user_key = f"user:{api_key[:8]}"
            user_existing = _user_usage_store.get(
                user_key,
                {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            user_existing["prompt_tokens"] += prompt_tokens
            user_existing["completion_tokens"] += completion_tokens
            user_existing["total_tokens"] += prompt_tokens + completion_tokens
            user_existing["model"] = model
            _user_usage_store[user_key] = user_existing


def pop_usage(request_id: str) -> dict | None:
    with _lock:
        return _usage_store.pop(request_id, None)


def get_user_usage(api_key: str | None = None) -> dict | None:
    if not api_key:
        return None
    with _lock:
        return _user_usage_store.get(f"user:{api_key[:8]}")
