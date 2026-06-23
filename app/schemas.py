from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AIRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Input text for processing")
    session_id: str = "default_session"
    task_type: str = "chat"

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Text to summarize")
    session_id: str = Field("default_session", description="Session ID for rate limiting")

class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Unstructured text to parse")
    schema_hint: Optional[str] = Field(None, max_length=500, description="Optional hint about the desired output structure")
    session_id: str = Field("default_session", description="Session ID for rate limiting")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_phrases: List[str] = Field(..., max_length=20)

class TaskType(str, Enum):
    SUMMARIZE = "summarize"
    PARSE = "parse"
    AGENT = "agent"
    PROCESS = "process"

class GatewayRequest(BaseModel):
    task_type: TaskType
    text: str = Field(..., min_length=1, max_length=50000, description="Input text")
    session_id: str = Field("default_session", description="Session ID for conversation continuity")
    stream: bool = Field(True, description="If true, returns SSE streaming response")
    schema_hint: Optional[str] = Field(None, max_length=500, description="Hint for parse task output structure")

class ChatStreamRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1, description="List of chat messages")
    user_id: str = Field("anonymous", description="User identifier")
    use_rag: bool = Field(False, description="Whether to augment with RAG context")

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")
    user_id: str = Field("anonymous", description="User identifier")

class ExtractRequest(BaseModel):
    file_path: str = Field(..., min_length=1, description="Path to the file to extract")
    file_type: str = Field(..., min_length=1, description="File type extension (txt, pdf, docx, png, jpg)")
    user_id: str = Field("anonymous", description="User identifier")

class RagIngestRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User identifier")
    document_id: str = Field(..., min_length=1, description="Document identifier")
    text: str = Field(..., min_length=1, description="Text content to ingest")

class RagRetrieveRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User identifier")
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
