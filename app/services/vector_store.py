import os
import uuid

_model = None
_chroma_client = None


def _cleanup():
    global _model, _chroma_client
    _chroma_client = None
    _model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db"))
        _chroma_client = chromadb.PersistentClient(path=db_path)
    return _chroma_client


def _get_collection(name: str = "knowledge_base"):
    return _get_client().get_or_create_collection(name=name)


collection = _get_collection()


def add_to_memory(text: str):
    doc_id = str(uuid.uuid4())
    col = _get_collection()
    col.add(documents=[text], ids=[doc_id])

def query_memory(query: str, n_results: int = 3) -> str:
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=n_results)
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return "No relevant context found."

def get_all_documents() -> list[str]:
    col = _get_collection()
    results = col.get()
    return results['documents'] if results and results['documents'] else []

def get_embedding(text: str) -> list[float]:
    return _get_model().encode(text).tolist()

def ingest_chunks(
    chunks: list[str],
    user_id: str,
    document_id: str,
    collection_name: str = "knowledge_base",
) -> int:
    col = _get_collection(collection_name)
    ids = [f"{document_id}__{i}" for i in range(len(chunks))]
    metadatas = [
        {"user_id": user_id, "document_id": document_id, "chunk_index": i}
        for i in range(len(chunks))
    ]
    col.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)

def retrieve_chunks(
    query: str,
    user_id: str,
    top_k: int = 5,
    collection_name: str = "knowledge_base",
) -> list[str]:
    col = _get_collection(collection_name)
    results = col.query(
        query_texts=[query],
        n_results=top_k,
        where={"user_id": user_id},
    )
    if results['documents'] and results['documents'][0]:
        return results['documents'][0]
    return []

def delete_document_chunks(
    document_id: str,
    user_id: str,
    collection_name: str = "knowledge_base",
) -> int:
    col = _get_collection(collection_name)
    results = col.get(
        where={"$and": [{"document_id": document_id}, {"user_id": user_id}]},
    )
    ids = results.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        return []
    overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
    if overlap < 0:
        overlap = 0
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks
