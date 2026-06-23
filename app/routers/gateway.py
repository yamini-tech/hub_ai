import asyncio
import json
import litellm
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from app.schemas import GatewayRequest, TaskType
from app.services.llm_client import call_llm_stream, tools
from app.services.memory_manager import read_knowledge_base
from app.services.search_service import web_search
from app.services.throttling import is_request_allowed, check_rate_limit
from app.services.job_manager import create_job
from app.services.task_processor import run_llm_task
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.core.security import verify_api_key
from app.core.config import API_KEY_ENABLED, MODEL_NAME, API_BASE, MAX_TOKENS_GATEWAY

router = APIRouter()
deps = [Depends(verify_api_key)] if API_KEY_ENABLED else []

@router.post("/ai/gateway", dependencies=deps)
async def ai_gateway(request: GatewayRequest, background_tasks: BackgroundTasks):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_GATEWAY)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit(request.session_id)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    if request.task_type == TaskType.SUMMARIZE:
        return await _handle_summarize(request)
    elif request.task_type == TaskType.PARSE:
        return await _handle_parse(request)
    elif request.task_type == TaskType.AGENT:
        return await _handle_agent(request)
    elif request.task_type == TaskType.PROCESS:
        return await _handle_process(request, background_tasks)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task_type: {request.task_type}")


async def _handle_summarize(request: GatewayRequest):
    model = select_model("summarize", request.text)
    system_prompt = get_system_prompt("summarize")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following text concisively:\n\n{request.text}"},
    ]
    if request.stream:
        return StreamingResponse(_stream(messages, model), media_type="text/event-stream")
    res = await call_llm_stream(messages, model=model)
    content = ""
    async for chunk in res:
        if c := chunk.choices[0].delta.content:
            content += c
    return {"summary": content}

async def _handle_parse(request: GatewayRequest):
    model = select_model("parse", request.text)
    system_prompt = get_system_prompt("parse")
    hint = f"\nDesired structure hint: {request.schema_hint}" if request.schema_hint else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Parse the following unstructured text into structured data:{hint}\n\n{request.text}"},
    ]
    if request.stream:
        return StreamingResponse(_stream(messages, model), media_type="text/event-stream")
    res = await call_llm_stream(messages, model=model)
    content = ""
    async for chunk in res:
        if c := chunk.choices[0].delta.content:
            content += c
    return {"parsed": content}

async def _handle_agent(request: GatewayRequest):
    model = select_model("agent", request.text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    system_prompt = get_system_prompt("agent", context=context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]
    if request.stream:
        return StreamingResponse(
            _agent_stream(messages, request.text, model),
            media_type="text/event-stream",
        )
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return {"answer": response.choices[0].message.content}

async def _handle_process(request: GatewayRequest, background_tasks: BackgroundTasks):
    job_id = create_job()
    background_tasks.add_task(
        run_llm_task, job_id, request.task_type.value, request.text, request.session_id
    )
    return {"job_id": job_id, "status": "processing"}


async def _stream(messages: list, model: str = ""):
    stream = await call_llm_stream(messages, model=model or None)
    try:
        async with asyncio.timeout(120):
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield f"data: {content}\n\n"
    except TimeoutError:
        yield "data: [STREAM_TIMEOUT]\n\n"
    yield "data: [DONE]\n\n"

async def _agent_stream(messages: list, original_text: str, model: str):
    stream = await call_llm_stream(messages, tools_list=tools, model=model)
    full_content = ""
    tool_calls_buffer = []
    try:
        async with asyncio.timeout(120):
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield f"data: {delta.content}\n\n"
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if len(tool_calls_buffer) <= tc.index:
                            tool_calls_buffer.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if tc.id:
                            tool_calls_buffer[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[tc.index]["function"]["arguments"] += tc.function.arguments
    except TimeoutError:
        yield "data: [STREAM_TIMEOUT]\n\n"
        return
    if tool_calls_buffer:
        yield f"data: __tool_calls__:{json.dumps(tool_calls_buffer)}\n\n"
        for tc in tool_calls_buffer:
            args = json.loads(tc["function"]["arguments"])
            tool_result = ""
            if tc["function"]["name"] == "web_search":
                tool_result = await web_search(args["query"])
            elif tc["function"]["name"] == "read_knowledge_base":
                tool_result = await asyncio.to_thread(read_knowledge_base, query=original_text)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(tool_result)})
        stream2 = await call_llm_stream(messages, model=model)
        try:
            async with asyncio.timeout(120):
                async for chunk in stream2:
                    if content := chunk.choices[0].delta.content:
                        yield f"data: {content}\n\n"
        except TimeoutError:
            yield "data: [STREAM_TIMEOUT]\n\n"
    yield "data: [DONE]\n\n"
