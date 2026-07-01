import asyncio
import json

from ddgs import DDGS


async def web_search(query: str) -> str:
    try:
        loop = asyncio.get_running_loop()

        def _search():
            results = list(DDGS().text(query, max_results=3))
            return json.dumps(results) if results else "[]"

        return await loop.run_in_executor(None, _search)
    except Exception:
        return "[Web search temporarily unavailable]"
