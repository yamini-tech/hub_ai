import logging
import time
from fastapi import Request

logger = logging.getLogger("smartbrain")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(_handler)


async def ai_usage_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        usage = getattr(request.state, "ai_usage", None)
        parts = [
            f"method={request.method}",
            f"path={request.url.path}",
            f"status={response.status_code}",
            f"latency_ms={latency_ms:.1f}",
        ]
        if usage:
            parts.append(f"model={getattr(request.state, 'model_used', 'unknown')}")
            parts.append(f"prompt_tokens={usage.prompt_tokens}")
            parts.append(f"completion_tokens={usage.completion_tokens}")
            parts.append(f"total_tokens={usage.total_tokens}")
        logger.info("  ".join(parts))
        return response
    except Exception:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "method=%s path=%s status=500 latency_ms=%.1f",
            request.method, request.url.path, latency_ms,
        )
        raise
