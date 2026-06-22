import asyncio
import json
import litellm
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.schemas import AIRequest
from app.services.llm_client import call_llm_stream, tools
from app.services.memory_manager import read_knowledge_base, save_to_knowledge_base, summarize_knowledge
from app.services.search_service import web_search
from app.services.task_processor import run_llm_task
from app.services.throttling import is_request_allowed, check_rate_limit
from app.services.job_manager import create_job, get_job
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.core.config import MODEL_NAME, API_BASE

router = APIRouter()
chat_histories: dict[str, list[dict]] = {}

@router.post("/agent")
async def agent_endpoint(request: AIRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit(request.session_id)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    text = request.text.strip()

    if text.lower().startswith("remember"):
        content_to_save = text.lower().replace("remember", "").strip()
        await asyncio.to_thread(save_to_knowledge_base, content_to_save)
        return {"answer": "I have stored that in my memory."}

    model = select_model("agent", text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    system_prompt = get_system_prompt("agent", context=context)

    session_id = request.session_id
    if session_id not in chat_histories:
        chat_histories[session_id] = [{"role": "system", "content": system_prompt}]
    chat_histories[session_id][0]["content"] = system_prompt
    messages = chat_histories[session_id]
    messages.append({"role": "user", "content": request.text})

    async def generate():
        stream = await call_llm_stream(messages, tools_list=tools, model=model)
        full_content = ""
        tool_calls_buffer = []
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
        if tool_calls_buffer:
            yield f"data: __tool_calls__:{json.dumps(tool_calls_buffer)}\n\n"
            for tc in tool_calls_buffer:
                args = json.loads(tc["function"]["arguments"])
                tool_result = ""
                if tc["function"]["name"] == "web_search":
                    tool_result = await web_search(args["query"])
                elif tc["function"]["name"] == "read_knowledge_base":
                    tool_result = await asyncio.to_thread(read_knowledge_base, query=request.text)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(tool_result)})
            stream2 = await call_llm_stream(messages, model=model)
            async for chunk in stream2:
                if content := chunk.choices[0].delta.content:
                    yield f"data: {content}\n\n"
        messages.append({"role": "assistant", "content": full_content})
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/agent/sync")
async def agent_sync(request: AIRequest):
    allowed, count = is_request_allowed(request.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit(request.session_id)
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

    model = select_model("agent", text)
    context = await asyncio.to_thread(read_knowledge_base, request.text)
    system_prompt = get_system_prompt("agent", context=context)

    if session_id not in chat_histories:
        chat_histories[session_id] = [{"role": "system", "content": system_prompt}]
    chat_histories[session_id][0]["content"] = system_prompt
    messages = chat_histories[session_id]
    messages.append({"role": "user", "content": request.text})

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message
    messages.append(message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_result = ""
            if tool_name == "web_search":
                tool_result = await web_search(args["query"])
            elif tool_name == "read_knowledge_base":
                tool_result = await asyncio.to_thread(read_knowledge_base, query=request.text)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })
        final_response = await litellm.acompletion(
            model=model,
            messages=messages
        )
        final_answer = final_response.choices[0].message.content
        messages.append({"role": "assistant", "content": final_answer})
        return {"answer": final_answer}

    messages.append({"role": "assistant", "content": message.content})
    return {"answer": message.content}

@router.post("/ai/process", status_code=202)
async def process_request(ai_req: AIRequest, background_tasks: BackgroundTasks):
    allowed, count = is_request_allowed(ai_req.text, max_tokens=10000)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Input exceeds token limit: {count} tokens.")
    rate_ok, req_count = check_rate_limit(ai_req.session_id)
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
