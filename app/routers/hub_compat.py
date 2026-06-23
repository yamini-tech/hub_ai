import asyncio
import json
from fastapi import APIRouter, HTTPException, Response, Depends
from fastapi.responses import StreamingResponse
from app.schemas import ChatStreamRequest, EmbedRequest, ExtractRequest, RagIngestRequest, RagRetrieveRequest
from app.services.llm_client import call_llm_stream
from app.services.memory_manager import read_knowledge_base
from app.services.vector_store import (
    get_embedding,
    ingest_chunks,
    retrieve_chunks,
    delete_document_chunks,
    chunk_text,
)
from app.services.document_extractor import extract_text
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.core.security import verify_api_key
from app.core.config import API_KEY_ENABLED, RATE_LIMIT_WINDOW_SEC, RATE_LIMIT_MAX_REQUESTS
from app.services.throttling import check_rate_limit

router = APIRouter()
_hub_deps = [Depends(verify_api_key)] if API_KEY_ENABLED else []


def _rl_key(user_id: str, endpoint: str) -> str:
    return f"hub:{endpoint}:{user_id or 'anonymous'}"


@router.post("/api/v1/chat/stream", dependencies=_hub_deps)
async def chat_stream(payload: ChatStreamRequest):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    rate_ok, req_count = check_rate_limit(_rl_key(payload.user_id, "chat_stream"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")

    last_user_msg = next(
        (m["content"] for m in reversed(payload.messages) if m["role"] == "user"),
        "",
    )
    model = select_model("chat", last_user_msg)

    messages = list(payload.messages)
    if payload.use_rag:
        context = await asyncio.to_thread(read_knowledge_base, last_user_msg)
        system_prompt = get_system_prompt("hub_chat", context=context)
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        system_prompt = get_system_prompt("chat")
        messages.insert(0, {"role": "system", "content": system_prompt})

    async def event_generator():
        stream = await call_llm_stream(messages, model=model)
        try:
            async with asyncio.timeout(120):
                async for chunk in stream:
                    if content := chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'delta': content})}\n\n"
        except TimeoutError:
            yield "data: [STREAM_TIMEOUT]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/embed", dependencies=_hub_deps)
async def embed_text(payload: EmbedRequest):
    rate_ok, req_count = check_rate_limit(_rl_key(payload.user_id, "embed"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    embedding = get_embedding(payload.text)
    return {"embedding": embedding}


@router.post("/api/v1/extract", dependencies=_hub_deps)
async def extract_document(payload: ExtractRequest):
    rate_ok, req_count = check_rate_limit(_rl_key(payload.user_id, "extract"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    try:
        text = extract_text(payload.file_path, payload.file_type)
        return {"text": text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {payload.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/api/v1/rag/ingest", dependencies=_hub_deps)
async def rag_ingest(payload: RagIngestRequest):
    rate_ok, req_count = check_rate_limit(_rl_key(payload.user_id, "rag_ingest"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    chunks = chunk_text(payload.text)
    count = ingest_chunks(chunks, payload.user_id, payload.document_id)
    return {"chunks_stored": count}


@router.post("/api/v1/rag/retrieve", dependencies=_hub_deps)
async def rag_retrieve(payload: RagRetrieveRequest):
    rate_ok, req_count = check_rate_limit(_rl_key(payload.user_id, "rag_retrieve"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    chunks = retrieve_chunks(payload.query, payload.user_id, top_k=payload.top_k)
    return {"chunks": chunks}


@router.delete("/api/v1/rag/documents/{document_id}", dependencies=_hub_deps)
async def rag_delete(document_id: str, user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    rate_ok, req_count = check_rate_limit(_rl_key(user_id, "rag_delete"), window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not rate_ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {req_count} requests in window.")
    count = delete_document_chunks(document_id, user_id)
    return Response(status_code=204)
