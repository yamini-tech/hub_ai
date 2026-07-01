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
from app.services.task_processor import run_llm_task
from app.services.throttling import check_rate_limit_by_user, is_request_allowed

router = APIRouter()
chat_histories: dict[str, list[dict]] = {}
MAX_HISTORY_PER_SESSION = 50
MAX_SESSIONS = 1000
_MAX_TOOL_DEPTH = 5


def _trim_session(session_id: str):
    messages = chat_histories.get(session_id)
    if not messages:
        return
    if len(messages) > MAX_HISTORY_PER_SESSION:
        keep = MAX_HISTORY_PER_SESSION // 2
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        chat_histories[session_id] = system_msgs + other_msgs[-keep:]


def _evict_sessions():
    if len(chat_histories) > MAX_SESSIONS:
        excess = len(chat_histories) - MAX_SESSIONS
        for key in list(chat_histories.keys())[:excess]:
            del chat_histories[key]


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
    if context and context not in ("No relevant context found.", "Knowledge base is empty."):
        system_prompt = get_system_prompt("agent", context=context)
    else:
        system_prompt = get_system_prompt("agent")

    session_id = request.session_id
    if session_id not in chat_histories:
        _evict_sessions()
        chat_histories[session_id] = [{"role": "system", "content": system_prompt}]
    chat_histories[session_id][0]["content"] = system_prompt
    messages = chat_histories[session_id]
    messages.append({"role": "user", "content": request.text})
    _trim_session(session_id)

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
                            yield f"data: {json.dumps({'delta': delta.content})}\n\n"
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
            tools_executed = 0
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
                tools_executed += 1
            if tools_executed == 0:
                break
        else:
            yield 'data: {"error": "max_tool_depth_exceeded"}\n\n'
            return

        messages.append({"role": "assistant", "content": full_content})
        _trim_session(session_id)
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
    if context and context not in ("No relevant context found.", "Knowledge base is empty."):
        system_prompt = get_system_prompt("agent", context=context)
    else:
        system_prompt = get_system_prompt("agent")

    if session_id not in chat_histories:
        _evict_sessions()
        chat_histories[session_id] = [{"role": "system", "content": system_prompt}]
    chat_histories[session_id][0]["content"] = system_prompt
    messages = chat_histories[session_id]
    messages.append({"role": "user", "content": request.text})
    _trim_session(session_id)

    for turn in range(_MAX_TOOL_DEPTH):
        message = await call_llm(messages, model=model, tools_list=tools)
        messages.append(message)

        if not message.tool_calls:
            final_answer = message.content or ""
            messages.append({"role": "assistant", "content": final_answer})
            _trim_session(session_id)
            if request.cross_session and final_answer:
                exchange = f"[session:{session_id}] User: {request.text[:200]} | Assistant: {final_answer[:500]}"
                await asyncio.to_thread(save_to_knowledge_base, exchange)
            return {"answer": final_answer}

        tools_executed = 0
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
            tools_executed += 1
        if tools_executed == 0:
            break
    else:
        raise HTTPException(status_code=500, detail="Agent exceeded max tool call depth")

    # tools_executed was 0 — use last message content as final answer
    _trim_session(session_id)
    if request.cross_session and message.content:
        exchange = f"[session:{session_id}] User: {request.text[:200]} | Assistant: {message.content[:500]}"
        await asyncio.to_thread(save_to_knowledge_base, exchange)
    return {"answer": message.content or ""}


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
