from fastapi import Request
import time
import uuid

class AIUsageLog(dict):
    pass

async def ai_usage_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        usage = getattr(request.state, "ai_usage", None)
        if usage:
            log_entry = {
                "request_id": str(uuid.uuid4()),
                "user_id": "anonymous",
                "task_type": "ai_process",
                "model_used": getattr(request.state, "model_used", "unknown"),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost": 0.0,
                "latency_ms": latency_ms,
            }
        return response
    except Exception as e:
        raise e
