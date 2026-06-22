from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AIRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Input text for processing")
    session_id: str = "default_session"
    task_type: str = "chat"

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Text to summarize")

class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="Unstructured text to parse")
    schema_hint: Optional[str] = Field(None, max_length=500, description="Optional hint about the desired output structure")

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
