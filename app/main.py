import asyncio
import os

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.core.config import API_KEY_ENABLED, poll_config, reload_config
from app.core.security import verify_api_key
from app.middleware.logging import ai_usage_middleware
from app.routers.chat import router as chat_router
from app.routers.gateway import router as gateway_router
from app.routers.hub_compat import router as hub_compat_router
from app.routers.tasks import router as tasks_router
from app.services.prompt_manager import _PROMPTS, load_prompts

MAX_REQUEST_SIZE = int(os.getenv("SMARTHUB_MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))
CORS_ORIGINS = os.getenv("SMARTHUB_CORS_ORIGINS", "")

app = FastAPI(
    title="SmartHub AI",
    description="Unified AI microservice for SmartHub — summarization, parsing, "
    "chat, and agent capabilities with streaming.",
    version="2.0.0",
)

app.middleware("http")(ai_usage_middleware)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in CORS_ORIGINS.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request too large. Maximum allowed size is {MAX_REQUEST_SIZE} bytes."},
        )
    return await call_next(request)


deps = [Depends(verify_api_key)] if API_KEY_ENABLED else []

app.include_router(chat_router, prefix="/api", tags=["Agent"], dependencies=deps)

# AI service endpoints — match the spec: POST /api/ai/summarize, POST /api/ai/parse
app.include_router(tasks_router, prefix="/api/ai", tags=["AI Services"], dependencies=deps)

# Centralized Gateway — single entry point for all AI operations
app.include_router(gateway_router, prefix="/api", tags=["Gateway"])

# SmartHub Backend Compatibility — endpoints hub_backend expects
app.include_router(hub_compat_router, tags=["Hub Compat"])


@app.get("/")
async def root():
    return {
        "service": "SmartHub AI Brain",
        "version": "2.0.0",
        "status": "online",
        "gateway": "POST /api/ai/gateway",
        "hub_compat": "POST /api/v1/chat/stream — hub_backend expects this",
        "docs": "/docs",
        "endpoints": {
            "health": "/health — liveness check",
            "ready": "/ready — readiness check (prompts loaded)",
            "gateway": "/api/ai/gateway — single entry point for all AI tasks",
            "hub_compat": "/api/v1/chat/stream — SmartHub backend streaming chat",
            "summarize": "/api/ai/summarize — streaming summarization",
            "parse": "/api/ai/parse — streaming unstructured data parsing",
            "agent": "/api/agent — streaming agent with tools and memory",
            "process": "/api/ai/process — async background processing",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    prompts_ok = bool(_PROMPTS)
    return {
        "status": "ready" if prompts_ok else "not ready",
        "prompts_loaded": prompts_ok,
    }


@app.on_event("startup")
async def startup_event():
    load_prompts()
    asyncio.create_task(_config_poll_loop())
    print("--- SmartHub Intelligence Brain is Online ---")
    for route in app.routes:
        if isinstance(route, APIRoute):
            print(f"  {list(route.methods)} {route.path}")


async def _config_poll_loop():
    """Background task that polls config.yaml for changes every
    POLL_INTERVAL_SEC seconds and hot-reloads model routing."""
    from app.services.config_watcher import POLL_INTERVAL_SEC

    while True:
        await asyncio.sleep(POLL_INTERVAL_SEC)
        try:
            poll_config()
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown_event():
    from app.services.vector_store import _cleanup

    _cleanup()
    print("--- SmartHub Intelligence Brain shut down gracefully ---")


@app.post("/admin/config/reload", dependencies=deps)
async def admin_reload_config():
    """Force a hot-reload of config.yaml without restarting the service."""
    info = reload_config()
    return {"message": "Configuration reloaded", **info}


@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred in the AI Brain", "details": str(exc)},
    )
