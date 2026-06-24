import logging
import time
import uuid
from fastapi import Request
from app.services.pricing import calculate_cost
from app.services.usage_tracker import set_request_id, pop_usage, set_api_key
from app.services.pii_redactor import redact_pii
from app.services.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    llm_requests_total,
    llm_tokens_total,
)

logger = logging.getLogger("smartbrain")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(_handler)


async def ai_usage_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    api_key = request.headers.get("X-API-KEY")
    set_api_key(api_key)
    start_time = time.time()

    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        usage = pop_usage(request_id)

        http_requests_total.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
        http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(latency_ms / 1000)

        safe_path = redact_pii(request.url.path)
        parts = [
            f"request_id={request_id}",
            f"method={request.method}",
            f"path={safe_path}",
            f"status={response.status_code}",
            f"latency_ms={latency_ms:.1f}",
        ]

        if usage:
            model = usage.get("model", "unknown")
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", 0)
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

            parts.append(f"model={model}")
            if isinstance(prompt_tokens, int):
                parts.append(f"prompt_tokens={prompt_tokens}")
                parts.append(f"completion_tokens={completion_tokens}")
                parts.append(f"total_tokens={total}")
                cost = calculate_cost(model, prompt_tokens, completion_tokens)
                parts.append(f"cost=${cost:.6f}")

            llm_requests_total.labels(model=model, status="ok").inc()
            if prompt_tokens:
                llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
            if completion_tokens:
                llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)

        logger.info("  ".join(parts))
        return response

    except Exception:
        latency_ms = (time.time() - start_time) * 1000
        http_requests_total.labels(method=request.method, path=request.url.path, status=500).inc()
        http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(latency_ms / 1000)
        logger.error(
            "request_id=%s method=%s path=%s status=500 latency_ms=%.1f",
            request_id, request.method, request.url.path, latency_ms,
        )
        raise
