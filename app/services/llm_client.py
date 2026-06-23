import asyncio
import litellm
from httpx import TimeoutException, ConnectError
from fastapi import HTTPException
from app.services.usage_tracker import record_llm_usage

RETRY_MAX = 3
RETRY_BASE_DELAY = 1.0

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


async def call_llm(messages: list, tools_list=None, response_format=None, model=None):
    try:
        response = await _call_with_retry(
            lambda: litellm.acompletion(
                model=model or "ollama/llama3.2",
                messages=messages,
                tools=tools_list,
                response_format=response_format,
            )
        )
        model_used = model or "ollama/llama3.2"
        try:
            r_usage = response.usage
            pt = int(getattr(r_usage, "prompt_tokens", 0))
            ct = int(getattr(r_usage, "completion_tokens", 0))
            if pt or ct:
                record_llm_usage(model=model_used, prompt_tokens=pt, completion_tokens=ct)
        except (TypeError, ValueError, AttributeError):
            pass
        return response.choices[0].message
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def call_llm_stream(messages: list, tools_list=None, model=None):
    try:
        response = await _call_with_retry(
            lambda: litellm.acompletion(
                model=model or "ollama/llama3.2",
                messages=messages,
                tools=tools_list,
                stream=True,
            )
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


