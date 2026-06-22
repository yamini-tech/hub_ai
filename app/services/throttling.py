import time
import tiktoken
from collections import defaultdict

# Per-session rate tracking
_rate_windows: dict[str, list[float]] = defaultdict(list)

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
    now = time.time()
    window_start = now - window_sec
    timestamps = _rate_windows[session_id]
    # Prune old entries
    _rate_windows[session_id] = [t for t in timestamps if t > window_start]
    count = len(_rate_windows[session_id])
    if count >= max_requests:
        return False, count
    _rate_windows[session_id].append(now)
    return True, count + 1
