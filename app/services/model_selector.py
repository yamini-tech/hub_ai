import os
from app.core.config import MODEL_NAME, TASK_MODEL_MAP, MODEL_MAP
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
    if task_type in TASK_MODEL_MAP:
        base = TASK_MODEL_MAP[task_type]
    else:
        base = MODEL_NAME

    if task_type in ("chat", "agent"):
        token_count = get_token_count(text, base)
        if token_count > _TOKEN_THRESHOLD:
            reasoner = MODEL_MAP.get("smart-hub-reasoner")
            if reasoner:
                return reasoner

    return base
