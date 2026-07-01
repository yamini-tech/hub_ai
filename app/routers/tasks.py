from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import MAX_TOKENS_TASK, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SEC
from app.schemas import ParseRequest, SummarizeRequest
from app.services.json_validator import validate_json_output
from app.services.llm_client import call_llm, call_llm_stream
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.services.throttling import check_rate_limit_by_user, is_request_allowed
from app.utils import _stream_sse

router = APIRouter()

RETRY_PARSE_MAX = 2


def _build_parse_messages(text: str, schema_hint: str | None = None, json_schema: dict | None = None) -> list[dict]:
    system_prompt = get_system_prompt("parse")
    hint = f"\nDesired output structure hint: {schema_hint}" if schema_hint else ""
    schema_instruction = ""
    if json_schema:
        import json as _json

        schema_instruction = f"\nThe response MUST conform to this JSON Schema:\n{_json.dumps(json_schema, indent=2)}"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Parse the following unstructured text into structured data:" f"{hint}{schema_instruction}\n\n{text}"
            ),
        },
    ]
    return messages


def _get_response_format(json_schema: dict | None = None) -> dict | None:
    if json_schema:
        return {"type": "json_schema", "json_schema": {"name": "parsed_output", "schema": json_schema, "strict": True}}
    return {"type": "json_object"}


@router.post("/summarize")
async def summarize(request: SummarizeRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_TASK)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("summarize", request.text)
    system_prompt = get_system_prompt("summarize")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following text concisively:\n\n{request.text}"},
    ]

    async def generate():
        stream = await call_llm_stream(messages, model=model)
        async for chunk in _stream_sse(stream):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/summarize/sync")
async def summarize_sync(request: SummarizeRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_TASK)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("summarize", request.text)
    system_prompt = get_system_prompt("summarize")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following text concisively:\n\n{request.text}"},
    ]
    res = await call_llm(messages, model=model)
    return {"summary": res.content}


@router.post("/parse")
async def parse_unstructured(request: ParseRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_TASK)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("parse", request.text)
    messages = _build_parse_messages(request.text, request.schema_hint, request.json_schema)
    response_format = _get_response_format(request.json_schema)

    async def generate():
        stream = await call_llm_stream(messages, model=model, response_format=response_format)
        async for chunk in _stream_sse(stream):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/parse/sync")
async def parse_unstructured_sync(request: ParseRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_TASK)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("parse", request.text)
    messages = _build_parse_messages(request.text, request.schema_hint, request.json_schema)
    response_format = _get_response_format(request.json_schema)

    for attempt in range(RETRY_PARSE_MAX):
        res = await call_llm(messages, model=model, response_format=response_format)
        content = res.content
        if not content:
            raise HTTPException(status_code=500, detail="LLM returned empty response")

        valid, error, parsed = validate_json_output(content, schema=request.json_schema)
        if valid:
            return {"parsed": parsed}

        if attempt < RETRY_PARSE_MAX - 1:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous output was not valid JSON. Error: {error}. " "Please return ONLY valid JSON."
                    ),
                }
            )

    raise HTTPException(
        status_code=422,
        detail=f"LLM failed to return valid structured data after {RETRY_PARSE_MAX} attempts. Last error: {error}",
    )
