import json
import asyncio
from duckduckgo_search import DDGS

async def web_search(query: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        def _search():
            with DDGS() as ddgs:
                return json.dumps(list(ddgs.text(query, max_results=3)))
        return await loop.run_in_executor(None, _search)
    except Exception:
        return "[Web search temporarily unavailable]"
