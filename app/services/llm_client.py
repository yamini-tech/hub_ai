import asyncio
import litellm
from httpx import TimeoutException, ConnectError
from fastapi import HTTPException
from app.services.usage_tracker import record_llm_usage
from app.services.cache import get_cache, set_cache
from app.services.pii_redactor import redact_pii, redact_stream
from app.services.injection_detector import check_injection
from app.services.content_filter import check_content, filter_stream, _BLOCKED_RESPONSE
from app.core.config import FALLBACK_CHAINS

RETRY_MAX = 3
RETRY_BASE_DELAY = 1.0

DEFAULT_MODEL = "ollama/llama3.2"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge_base",
            "description": "Read the saved research notes from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }
]

_RETRYABLE = (TimeoutException, ConnectError, litellm.RateLimitError)


def _get_fallback_models(model: str) -> list[str]:
    for prefix, fallbacks in FALLBACK_CHAINS.items():
        if model.startswith(prefix) or model.endswith(prefix):
            return fallbacks
    return []


async def _call_with_retry(coro_factory):
    last_exc = None
    for attempt in range(RETRY_MAX):
        try:
            return await coro_factory()
        except _RETRYABLE as e:
            last_exc = e
            if attempt < RETRY_MAX - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        except Exception:
            raise
    raise HTTPException(
        status_code=503,
        detail=f"LLM service unavailable after {RETRY_MAX} retries: {last_exc}",
    )


async def _call_with_fallback(coro_factory, model: str):
    models_to_try = [model] + _get_fallback_models(model)
    if models_to_try and not models_to_try[-1]:
        models_to_try = models_to_try[:-1]
    if not models_to_try:
        models_to_try = [DEFAULT_MODEL]

    last_exc = None
    for fallback_idx, m in enumerate(models_to_try):
        try:
            response = await _call_with_retry(lambda: coro_factory(m))
            return response, m
        except HTTPException as e:
            if e.status_code == 503 and fallback_idx < len(models_to_try) - 1:
                last_exc = e
                continue
            raise
    raise HTTPException(
        status_code=503,
        detail=f"All fallback models exhausted: {last_exc}",
    )


async def call_llm(messages: list, tools_list=None, response_format=None, model=None):
    model = model or DEFAULT_MODEL
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            injection = check_injection(m["content"])
            if injection:
                raise HTTPException(status_code=400, detail=injection)
    cached = get_cache(messages, model, tools_list, response_format)
    if cached is not None:
        if cached.content:
            cached.content = redact_pii(cached.content)
        return cached
    try:
        response, model_used = await _call_with_fallback(
            lambda m: litellm.acompletion(
                model=m,
                messages=messages,
                tools=tools_list,
                response_format=response_format,
            ),
            model=model,
        )
        try:
            r_usage = response.usage
            pt = int(getattr(r_usage, "prompt_tokens", 0))
            ct = int(getattr(r_usage, "completion_tokens", 0))
            if pt or ct:
                record_llm_usage(model=model_used, prompt_tokens=pt, completion_tokens=ct)
        except (TypeError, ValueError, AttributeError):
            pass
        result = response.choices[0].message
        if result.content:
            result.content = redact_pii(result.content)
        if result.content:
            safe, reason = check_content(result.content)
            if not safe:
                result.content = _BLOCKED_RESPONSE
        set_cache(messages, model_used, result, tools_list, response_format)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def call_llm_stream(messages: list, tools_list=None, model=None):
    model = model or DEFAULT_MODEL
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            injection = check_injection(m["content"])
            if injection:
                raise HTTPException(status_code=400, detail=injection)
    try:
        response, _ = await _call_with_fallback(
            lambda m: litellm.acompletion(
                model=m,
                messages=messages,
                tools=tools_list,
                stream=True,
            ),
            model=model,
        )
        return filter_stream(redact_stream(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


