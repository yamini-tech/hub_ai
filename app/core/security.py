from fastapi import Security, HTTPException, status, Request
from fastapi.security.api_key import APIKeyHeader
from app.core.config import API_KEY, API_KEY_ENABLED

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
    return True
