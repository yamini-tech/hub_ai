from fastapi import Security, HTTPException, status, Request
from fastapi.security.api_key import APIKeyHeader
from app.core.config import API_KEY, API_KEY_ENABLED, RATE_LIMIT_WINDOW_SEC, RATE_LIMIT_MAX_REQUESTS
from app.services.throttling import check_rate_limit_by_key

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    if not API_KEY_ENABLED:
        return True
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )
    ok, count = check_rate_limit_by_key(api_key, window_sec=RATE_LIMIT_WINDOW_SEC, max_requests=RATE_LIMIT_MAX_REQUESTS)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"API key rate limit exceeded: {count} requests in window.",
        )
    return True
