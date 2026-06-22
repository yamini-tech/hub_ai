from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import SummarizeRequest, ParseRequest
from app.services.llm_client import call_llm, call_llm_stream
from app.services.throttling import is_request_allowed, check_rate_limit
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.core.config import RATE_LIMIT_WINDOW_SEC, RATE_LIMIT_MAX_REQUESTS

router = APIRouter()

@router.post("/summarize")
async def summarize(request: SummarizeRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=20000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit("tasks:summarize", window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
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
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/summarize/sync")
async def summarize_sync(request: SummarizeRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=20000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit("tasks:summarize", window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
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
    allowed, count = is_request_allowed(request.text, max_tokens=20000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit("tasks:parse", window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("parse", request.text)
    system_prompt = get_system_prompt("parse")
    hint = f"\nDesired output structure hint: {request.schema_hint}" if request.schema_hint else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Parse the following unstructured text into structured data:{hint}\n\n{request.text}"},
    ]

    async def generate():
        stream = await call_llm_stream(messages, model=model)
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/parse/sync")
async def parse_unstructured_sync(request: ParseRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=20000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit("tasks:parse", window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    model = select_model("parse", request.text)
    system_prompt = get_system_prompt("parse")
    hint = f"\nDesired output structure hint: {request.schema_hint}" if request.schema_hint else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Parse the following unstructured text into structured data:{hint}\n\n{request.text}"},
    ]
    res = await call_llm(messages, model=model)
    return {"parsed": res.content}
