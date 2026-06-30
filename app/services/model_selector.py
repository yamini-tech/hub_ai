import os
from app.services.config_watcher import get_config_watcher
from app.services.throttling import get_token_count

# Read token threshold (default: 500 tokens). Falls back to converting SMARTHUB_MODEL_THRESHOLD if provided.
_token_threshold_str = os.getenv("SMARTHUB_MODEL_TOKEN_THRESHOLD")
_char_threshold_str = os.getenv("SMARTHUB_MODEL_THRESHOLD")

_TEXT_LENGTH_THRESHOLD = int(_char_threshold_str) if _char_threshold_str is not None else 2000

if _token_threshold_str is not None:
    _TOKEN_THRESHOLD = int(_token_threshold_str)
else:
    _TOKEN_THRESHOLD = max(1, _TEXT_LENGTH_THRESHOLD // 4)


def select_model(task_type: str, text: str = "") -> str:
    watcher = get_config_watcher()
    base = watcher.get_task_model(task_type) or watcher.get_default_model()

    if task_type in ("chat", "agent"):
        token_count = get_token_count(text, base)
        if token_count > _TOKEN_THRESHOLD:
            reasoner = watcher.get_model("smart-hub-reasoner")
            if reasoner:
                return reasoner

    return base
