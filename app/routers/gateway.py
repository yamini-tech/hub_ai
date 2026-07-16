import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import (
    API_KEY_ENABLED,
    MAX_TOKENS_GATEWAY,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SEC,
)
from app.core.security import verify_api_key
from app.schemas import GatewayRequest, TaskType
from app.services.job_manager import create_job
from app.services.json_validator import validate_json_output
from app.services.llm_client import call_llm, call_llm_stream, tools
from app.services.memory_manager import read_knowledge_base
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.services.search_service import web_search
from app.services.session_manager import (
    append_assistant_message,
    build_llm_messages,
    get_or_create_session,
    store_session_fact,
)
from app.services.task_processor import run_llm_task
from app.services.throttling import check_rate_limit_by_user, is_request_allowed

router = APIRouter()
deps = [Depends(verify_api_key)] if API_KEY_ENABLED else []
_MAX_TOOL_DEPTH = 5


@router.post("/ai/gateway", dependencies=deps)
async def ai_gateway(request: GatewayRequest, background_tasks: BackgroundTasks):
    allowed, count = is_request_allowed(request.text, max_tokens=MAX_TOKENS_GATEWAY)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count}")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
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
    model = request.model or select_model("summarize", request.text)
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
    model = request.model or select_model("parse", request.text)
    system_prompt = get_system_prompt("parse")
    hint = f"\nDesired structure hint: {request.schema_hint}" if request.schema_hint else ""
    schema_instruction = ""
    if request.json_schema:
        schema_instruction = (
            f"\nThe response MUST conform to this JSON Schema:\n{json.dumps(request.json_schema, indent=2)}"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Parse the following unstructured text into structured data:"
                f"{hint}{schema_instruction}\n\n{request.text}"
            ),
        },
    ]
    if request.json_schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "parsed_output",
                "schema": request.json_schema,
                "strict": True,
            },
        }
    else:
        response_format = {"type": "json_object"}

    if request.stream:
        return StreamingResponse(
            _stream(messages, model, response_format=response_format),
            media_type="text/event-stream",
        )
    stream = await call_llm_stream(messages, model=model, response_format=response_format)
    content = ""
    async for chunk in stream:
        if c := chunk.choices[0].delta.content:
            content += c
    valid, error, parsed = validate_json_output(content, schema=request.json_schema)
    if not valid:
        raise HTTPException(status_code=422, detail=f"Parse output validation failed: {error}")
    return {"parsed": parsed}


async def _handle_agent(request: GatewayRequest):
    model = request.model or select_model("agent", request.text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    system_prompt = get_system_prompt("agent", context=context, session_context="")
    session_id = request.session_id
    get_or_create_session(session_id, system_prompt)
    messages = build_llm_messages(session_id, system_prompt, request.text)
    if request.stream:
        return StreamingResponse(
            _agent_stream(messages, request.text, model, session_id),
            media_type="text/event-stream",
        )
    for _ in range(_MAX_TOOL_DEPTH):
        message = await call_llm(messages, model=model, tools_list=tools)

        if not message.tool_calls:
            append_assistant_message(session_id, message.content or "")
            store_session_fact(session_id, request.text, message.content or "")
            return {"answer": message.content or ""}

        assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_msg)

        tool_results_text = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                continue
            tool_result = ""
            if tool_name == "web_search":
                tool_result = await web_search(args["query"])
            elif tool_name == "read_knowledge_base":
                tool_result = await asyncio.to_thread(read_knowledge_base, query=request.text)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)})
            tool_results_text.append(f"[{tool_name}]: {tool_result}")

        if tool_results_text:
            joined = "\n".join(tool_results_text)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Use the following information to answer the user's question directly and concisely. "
                        f"Do NOT mention tools, search, external resources, or how you obtained the information — "
                        f"just give the answer.\n\n{joined}"
                    ),
                }
            )

    final = await call_llm(messages, model=model)
    append_assistant_message(session_id, final.content or "")
    store_session_fact(session_id, request.text, final.content or "")
    return {"answer": final.content or ""}


async def _handle_process(request: GatewayRequest, background_tasks: BackgroundTasks):
    job_id = create_job()
    background_tasks.add_task(run_llm_task, job_id, request.task_type.value, request.text, request.session_id)
    return {"job_id": job_id, "status": "processing"}


async def _stream(messages: list, model: str = "", response_format=None):
    stream = await call_llm_stream(messages, model=model or None, response_format=response_format)
    try:
        async with asyncio.timeout(120):
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'token': content})}\n\n"
    except TimeoutError:
        yield 'data: {"error": "stream_timeout"}\n\n'
    yield "data: [DONE]\n\n"


async def _agent_stream(messages: list, original_text: str, model: str, session_id: str):
    stream = await call_llm_stream(messages, tools_list=tools, model=model)
    full_content = ""
    tool_calls_buffer = []
    try:
        async with asyncio.timeout(120):
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"
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
        yield 'data: {"error": "stream_timeout"}\n\n'
        return
    if tool_calls_buffer:
        yield f"data: __tool_calls__:{json.dumps(tool_calls_buffer)}\n\n"
        tool_results_text = []
        for tc in tool_calls_buffer:
            args = json.loads(tc["function"]["arguments"])
            tool_result = ""
            if tc["function"]["name"] == "web_search":
                tool_result = await web_search(args["query"])
            elif tc["function"]["name"] == "read_knowledge_base":
                tool_result = await asyncio.to_thread(read_knowledge_base, query=original_text)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(tool_result)})
            tool_results_text.append(f"[{tc['function']['name']}]: {tool_result}")
        if tool_results_text:
            joined = "\n".join(tool_results_text)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Use the following information to answer the user's question directly and concisely. "
                        f"Do NOT mention tools, search, external resources, or how you obtained the information — "
                        f"just give the answer.\n\n{joined}"
                    ),
                }
            )
        full_content = ""
        stream2 = await call_llm_stream(messages, model=model)
        try:
            async with asyncio.timeout(120):
                async for chunk in stream2:
                    if content := chunk.choices[0].delta.content:
                        full_content += content
                        yield f"data: {json.dumps({'token': content})}\n\n"
        except TimeoutError:
            yield 'data: {"error": "stream_timeout"}\n\n'
    append_assistant_message(session_id, full_content)
    store_session_fact(session_id, original_text, full_content)
    yield "data: [DONE]\n\n"
