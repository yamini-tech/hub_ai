import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SEC
from app.schemas import AIRequest
from app.services.job_manager import create_job, get_job
from app.services.llm_client import call_llm, call_llm_stream, tools
from app.services.memory_manager import read_knowledge_base, save_to_knowledge_base, summarize_knowledge
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
_MAX_TOOL_DEPTH = 5


@router.post("/agent")
async def agent_endpoint(request: AIRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    text = request.text.strip()

    if text.lower().startswith("remember"):
        content_to_save = text.lower().replace("remember", "").strip()
        await asyncio.to_thread(save_to_knowledge_base, content_to_save)
        return {"answer": "I have stored that in my memory."}

    model = request.model or select_model("agent", text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    if not context or context in ("No relevant context found.", "Knowledge base is empty."):
        context = ""
    system_prompt = get_system_prompt("agent", context=context, session_context="")

    session_id = request.session_id
    get_or_create_session(session_id, system_prompt)
    messages = build_llm_messages(session_id, system_prompt, request.text)

    async def generate():
        full_content = ""
        for turn in range(_MAX_TOOL_DEPTH):
            stream = await call_llm_stream(messages, tools_list=tools, model=model)
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

            if not tool_calls_buffer:
                break

            yield f"data: __tool_calls__:{json.dumps(tool_calls_buffer)}\n\n"
            tool_results_text = []
            for tc in tool_calls_buffer:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    continue
                tool_result = ""
                if tc["function"]["name"] == "web_search":
                    tool_result = await web_search(args["query"])
                elif tc["function"]["name"] == "read_knowledge_base":
                    tool_result = await asyncio.to_thread(read_knowledge_base, query=request.text)
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
        else:
            yield 'data: {"error": "max_tool_depth_exceeded"}\n\n'
            return

        append_assistant_message(session_id, full_content)
        store_session_fact(session_id, request.text, full_content)
        if request.cross_session and full_content:
            exchange = f"[session:{session_id}] User: {request.text[:200]} | Assistant: {full_content[:500]}"
            await asyncio.to_thread(save_to_knowledge_base, exchange)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent/sync")
async def agent_sync(request: AIRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit_by_user(
        request.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    text = request.text.strip()
    session_id = request.session_id

    if text.lower().startswith("remember"):
        content_to_save = text.lower().replace("remember", "").strip()
        await asyncio.to_thread(save_to_knowledge_base, content_to_save)
        return {"answer": "I have stored that in my memory."}

    if "summarize" in text.lower():
        result = await summarize_knowledge()
        return {"answer": result}

    model = request.model or select_model("agent", text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    if not context or context in ("No relevant context found.", "Knowledge base is empty."):
        context = ""
    system_prompt = get_system_prompt("agent", context=context, session_context="")

    get_or_create_session(session_id, system_prompt)
    messages = build_llm_messages(session_id, system_prompt, request.text)

    for turn in range(_MAX_TOOL_DEPTH):
        message = await call_llm(messages, model=model, tools_list=tools)

        assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            final_answer = message.content or ""
            append_assistant_message(session_id, final_answer)
            store_session_fact(session_id, request.text, final_answer)
            if request.cross_session and final_answer:
                exchange = f"[session:{session_id}] User: {request.text[:200]} | Assistant: {final_answer[:500]}"
                await asyncio.to_thread(save_to_knowledge_base, exchange)
            return {"answer": final_answer}

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
    final_answer = final.content or ""
    append_assistant_message(session_id, final_answer)
    store_session_fact(session_id, request.text, final_answer)
    if request.cross_session and final_answer:
        exchange = f"[session:{session_id}] User: {request.text[:200]} | Assistant: {final_answer[:500]}"
        await asyncio.to_thread(save_to_knowledge_base, exchange)
    return {"answer": final_answer}


@router.post("/ai/process", status_code=202)
async def process_request(ai_req: AIRequest, background_tasks: BackgroundTasks):
    allowed, count = is_request_allowed(ai_req.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit_by_user(
        ai_req.session_id,
        window_sec=RATE_LIMIT_WINDOW_SEC,
        max_requests=RATE_LIMIT_MAX_REQUESTS,
    )
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    job_id = create_job()
    background_tasks.add_task(run_llm_task, job_id, ai_req.task_type, ai_req.text, ai_req.session_id)
    return {"job_id": job_id, "status": "processing"}


@router.get("/ai/status/{job_id}")
async def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
