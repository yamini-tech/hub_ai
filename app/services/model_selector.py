import os
from app.core.config import MODEL_NAME, TASK_MODEL_MAP, MODEL_MAP

_TEXT_LENGTH_THRESHOLD = int(os.getenv("SMARTHUB_MODEL_THRESHOLD", "2000"))


def select_model(task_type: str, text: str = "") -> str:
    if task_type in TASK_MODEL_MAP:
        base = TASK_MODEL_MAP[task_type]
    else:
        base = MODEL_NAME

    if task_type in ("chat", "agent") and len(text) > _TEXT_LENGTH_THRESHOLD:
        reasoner = MODEL_MAP.get("smart-hub-reasoner")
        if reasoner:
            return reasoner

    return base
