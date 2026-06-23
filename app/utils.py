import asyncio


STREAM_TIMEOUT_SEC = 120


async def _stream_sse(stream, done_marker="data: [DONE]\n\n"):
    try:
        async with asyncio.timeout(STREAM_TIMEOUT_SEC):
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield f"data: {content}\n\n"
    except TimeoutError:
        yield "data: [STREAM_TIMEOUT]\n\n"
    yield done_marker
