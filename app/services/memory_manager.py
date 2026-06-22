import asyncio
import litellm
from app.core.config import MODEL_NAME, API_BASE
from app.services.vector_store import add_to_memory, query_memory, get_all_documents

def save_to_knowledge_base(content: str):
    add_to_memory(content)
    return "Saved Semantic Memory."

def read_knowledge_base(query: str = ""):
    if query:
        return query_memory(query)
    docs = get_all_documents()
    return "\n".join(docs) if docs else "Knowledge base is empty."

async def summarize_knowledge():
    content = await asyncio.to_thread(read_knowledge_base)
    if not content or content == "Knowledge base is empty.":
        return "Knowledge base is empty."
    response = await litellm.acompletion(
        model=MODEL_NAME,
        api_base=API_BASE,
        messages=[{"role": "user", "content": f"Summarize these notes:\n{content}"}]
    )
    return response.choices[0].message.content
