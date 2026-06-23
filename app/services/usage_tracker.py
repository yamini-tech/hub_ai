import contextvars
import threading

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")

_usage_store: dict[str, dict] = {}
_lock = threading.Lock()


def set_request_id(request_id: str):
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    try:
        return request_id_var.get()
    except LookupError:
        return None


def record_llm_usage(model: str, prompt_tokens: int, completion_tokens: int):
    rid = get_request_id()
    if not rid:
        return
    with _lock:
        existing = _usage_store.get(
            rid,
            {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        existing["prompt_tokens"] += prompt_tokens
        existing["completion_tokens"] += completion_tokens
        existing["total_tokens"] += prompt_tokens + completion_tokens
        existing["model"] = model
        _usage_store[rid] = existing


def pop_usage(request_id: str) -> dict | None:
    with _lock:
        return _usage_store.pop(request_id, None)
