# SmartHub AI Brain

A unified internal AI microservice for SmartHub. The backend sends a task type and text — the Brain handles model selection, prompt engineering, token limits, rate throttling, and response streaming.

## Quick Start

```bash
# 1. Create venv and install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with real API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

# 3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload

# 4. Verify
curl http://localhost:8003/
```

## How SmartHub Integrates (The Abstraction Layer)

SmartHub calls the Brain. The Brain calls the LLM. SmartHub never touches a model API.

```
                          ┌─────────────────────────────────────────┐
                          │           SmartHub AI Brain             │
                          │                                         │
SmartHub Backend          │  POST /api/ai/gateway                   │
  │                       │    {                                    │
  │  task_type + text     │      "task_type": "summarize",          │
  │────────────────────>  │      "text": "document content...",     │
  │                       │      "session_id": "user_abc"           │
  │  summary or stream    │    }                                    │
  │<────────────────────  │                                         │
  │                       │  1. select_model("summarize", text)     │
  │                       │     → "openai/gpt-4o"                   │
  │                       │  2. get_system_prompt("summarize")      │
  │                       │     → "You are a precise summarizer..." │
  │                       │  3. check_rate_limit("user_abc")        │
  │                       │     → 30 requests left this minute      │
  │                       │  4. is_request_allowed(text)            │
  │                       │     → 142 tokens (under 20000 limit)    │
  │                       │  5. call_llm_stream(messages, model)    │
  │                       │     → SSE stream or JSON response       │
  │                       └─────────────────────────────────────────┘
```

### Minimal Integration Example

```python
import httpx

response = httpx.post("http://localhost:8003/api/ai/gateway", json={
    "task_type": "summarize",
    "text": "Your document or text here...",
    "session_id": "optional-session-id",
    "stream": False,
})

print(response.json())  # {"summary": "..."}
```

**SmartHub never needs to know:**
- Which model is being used (OpenAI, Anthropic, HuggingFace, Ollama)
- How to construct a provider API call
- What system prompt to inject
- How to count tokens or check rate limits

## Endpoints

### `POST /api/ai/gateway` — Unified Gateway

The primary endpoint. Handles all task types in one place.

| Field | Type | Required | Description |
|---|---|---|---|
| `task_type` | `"summarize"`, `"parse"`, `"agent"`, `"process"` | yes | What the Brain should do |
| `text` | string (1–50000 chars) | yes | The input content |
| `session_id` | string | no | Used for rate limiting and chat history |
| `stream` | boolean | no (default: `true`) | If `true`, returns SSE stream |
| `schema_hint` | string | no | Hint for `parse` output structure |

**Response (sync):**
```json
{"summary": "..."}   // for summarize
{"parsed": "..."}    // for parse
{"answer": "..."}    // for agent
{"job_id": "...", "status": "processing"}  // for process
```

**Response (streaming):** SSE format:
```
data: {"delta": "The"}
data: {"delta": " summary"}
data: {"delta": " is..."}
data: [DONE]
```

### `POST /api/ai/summarize` | `POST /api/ai/summarize/sync`

Direct summarization — streaming and synchronous variants.

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string (1–100000 chars) | yes | Text to summarize |

### `POST /api/ai/parse` | `POST /api/ai/parse/sync`

Parse unstructured text into structured data.

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string (1–100000 chars) | yes | Unstructured text |
| `schema_hint` | string | no | Desired output structure hint |

### `POST /api/agent` | `POST /api/agent/sync`

Chatbot with tools (web search, knowledge base), memory, and streaming.

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string (1–50000 chars) | yes | User message |
| `session_id` | string | no (default: `"default_session"`) | Conversation thread |
| `task_type` | string | no (default: `"chat"`) | Task routing |

### Hub Backend Compatibility (`/api/v1/*`)

These endpoints match the contract hub_backend's `llm_service.py` expects.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat/stream` | SSE streaming chat (with optional RAG context) |
| `POST` | `/api/v1/embed` | Generate text embeddings (384-dim) |
| `POST` | `/api/v1/extract` | Extract text from files (txt, pdf, docx, png, jpg) |
| `POST` | `/api/v1/rag/ingest` | Store chunks in vector database |
| `POST` | `/api/v1/rag/retrieve` | Query vector database with user isolation |
| `DELETE` | `/api/v1/rag/documents/{document_id}` | Delete document vectors by user |

### `POST /api/ai/process` | `GET /api/ai/status/{job_id}`

Async background processing with job status polling.

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SMARTHUB_API_KEY` | (none) | If set, all backed endpoints require this key via `X-API-KEY` header |
| `SMARTHUB_MODEL` | `ollama/llama3.2` | Default LLM model |
| `SMARTHUB_API_BASE` | `http://localhost:11434` | API base for Ollama |
| `SMARTHUB_RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `SMARTHUB_RATE_LIMIT_MAX` | `30` | Max requests per window |
| `SMARTHUB_MODEL_THRESHOLD` | `2000` | Text length in chars above which chat routes to the reasoner model |
| `OPENAI_API_KEY` | (required for OpenAI) | OpenAI API key |
| `ANTHROPIC_API_KEY` | (required for Anthropic) | Anthropic API key |
| `HUGGINGFACE_API_KEY` | (required for HF TGI) | HuggingFace token |
| `HUGGINGFACE_API_BASE` | (required for HF TGI) | HuggingFace TGI endpoint |

### Model Config (`config/config.yaml`)

```yaml
model_list:
  - model_name: smart-hub-summarizer
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: smart-hub-parser
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: smart-hub-reasoner
    litellm_params:
      model: anthropic/claude-3-7-sonnet-20250219
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: smart-hub-hf-chat
    litellm_params:
      model: huggingface/meta-llama/Meta-Llama-3-8B-Instruct
      api_base: os.environ/HUGGINGFACE_API_BASE
      api_key: os.environ/HUGGINGFACE_API_KEY
```

### System Prompts (`config/system_prompts.yaml`)

```yaml
system_prompts:
  default: "You are a helpful AI assistant."
  summarize: "You are a precise summarizer..."
  parse: "You extract structured data from unstructured text..."
  chat: "You are a helpful assistant."
  agent: "You are an expert AI assistant. Answer using ONLY the provided CONTEXT below..."
```

## How Model Routing Works

The `model_selector` (`app/services/model_selector.py`) decides which LLM to use:

1. **By task type:** summarize → `openai/gpt-4o`, parse → `claude-3-5-sonnet`, chat → `huggingface/llama3`
2. **By text length:** If chat/agent input exceeds 2000 chars, routes to `claude-3-7-sonnet` (the reasoner)
3. **By explicit type:** `reasoning` task always routes to `claude-3-7-sonnet`
4. **Fallback:** `ollama/llama3.2` when no config entry matches

## Architecture

```
hub_ai/
├── app/
│   ├── main.py                  # FastAPI app assembly + startup
│   ├── schemas.py               # Pydantic models (AIRequest, GatewayRequest, etc.)
│   ├── core/
│   │   ├── config.py            # Env vars, model map, yaml config loader
│   │   └── security.py          # API key verification via X-API-KEY header
│   ├── routers/
│   │   ├── gateway.py           # POST /api/ai/gateway (unified entry point)
│   │   ├── tasks.py             # POST /api/ai/summarize, /parse
│   │   ├── chat.py              # POST /api/agent, /ai/process, job status
│   │   └── hub_compat.py        # /api/v1/* endpoints for hub_backend
│   ├── services/
│   │   ├── llm_client.py        # litellm wrapper (call_llm, call_llm_stream)
│   │   ├── model_selector.py    # Intelligent model routing by task & length
│   │   ├── prompt_manager.py    # System prompt loading and injection
│   │   ├── throttling.py        # Token counting + rate limiting
│   │   ├── vector_store.py      # ChromaDB: embed, ingest, retrieve, delete
│   │   ├── document_extractor.py# Text extraction (txt, pdf, docx, images)
│   │   ├── memory_manager.py    # Knowledge base read/write/summarize
│   │   ├── search_service.py    # Web search via DuckDuckGo
│   │   ├── embedding_service.py # Sentence-transformer embeddings
│   │   ├── job_manager.py       # Job queue (Redis with in-memory fallback)
│   │   ├── task_processor.py    # Async background task runner
│   │   └── pricing.py           # Cost calculation per model
│   └── middleware/
│       └── logging.py           # AI usage logging middleware
├── config/
│   ├── config.yaml              # Model definitions
│   └── system_prompts.yaml      # Prompt templates per task
├── chroma_db/                   # Vector database (auto-generated)
├── tests/                       # Test suite (152 tests, pytest)
│   ├── conftest.py              # Shared fixtures, mocks for heavy deps
│   ├── test_schemas.py          # Pydantic model validation tests
│   ├── test_throttling.py       # Token counting & rate limit tests
│   ├── test_prompt_manager.py   # Prompt loading & retrieval tests
│   ├── test_model_selector.py   # Model routing logic tests
│   ├── test_pricing.py          # Cost calculation tests
│   ├── test_security.py         # API key verification tests
│   ├── test_config.py           # Configuration loading tests
│   ├── test_job_manager.py      # Job & chat history CRUD tests
│   ├── test_vector_store.py     # Chunk text edge case tests
│   ├── test_document_extractor.py # Text extraction tests
│   ├── test_llm_client.py       # LLM wrapper (mock) tests
│   ├── test_main.py             # App root & exception handler tests
│   ├── test_tasks.py            # Summarize/parse endpoint tests
│   ├── test_chat.py             # Agent/process endpoint tests
│   ├── test_gateway.py          # Unified gateway endpoint tests
│   └── test_hub_compat.py       # Hub compat endpoint tests
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `litellm` | LLM abstraction (100+ providers) |
| `chromadb` | Vector database for RAG |
| `sentence-transformers` | Text embeddings (all-MiniLM-L6-v2) |
| `tiktoken` | Token counting |
| `PyMuPDF` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `Pillow` + `pytesseract` | Image OCR (requires `brew install tesseract`) |
| `duckduckgo_search` | Web search tool |
| `pyyaml` | Config file parsing |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Manual Testing

Once the server is running (`uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload`), you can test endpoints with `curl`.

### Without API Keys (no LLM calls)

These test the server infrastructure without needing any provider keys:

```bash
# 1. Root health check
curl http://localhost:8003/

# 2. Swagger UI — open in browser:
# http://localhost:8003/docs

# 3. Trigger token limit exceeded (returns 429)
curl -X POST http://localhost:8003/api/ai/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"'"$(python -c "print('x'*5000)")"'"}'

# 4. Async job creation (background task will fail, but job is created)
curl -X POST http://localhost:8003/api/ai/process \
  -H "Content-Type: application/json" \
  -d '{"text":"hello","task_type":"summarize"}'

# 5. Missing required fields (returns 422)
curl -X POST http://localhost:8003/api/ai/summarize \
  -H "Content-Type: application/json" \
  -d '{}'
```

### With a Local Ollama (free, no API key needed)

If you have Ollama installed and the `llama3.2` model pulled, LLM calls work immediately:

```bash
# 6. Summarize (sync)
curl -X POST http://localhost:8003/api/ai/summarize/sync \
  -H "Content-Type: application/json" \
  -d '{"text":"The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."}'
```

### With API Keys

Set one or more provider keys in `.env`, then:

```bash
# 7. Gateway — summarize (streaming SSE)
curl -X POST http://localhost:8003/api/ai/gateway \
  -H "Content-Type: application/json" \
  -d '{"task_type":"summarize","text":"Long document text here...","stream":true}'

# 8. Gateway — summarize (sync JSON)
curl -X POST http://localhost:8003/api/ai/gateway \
  -H "Content-Type: application/json" \
  -d '{"task_type":"summarize","text":"Long document text here...","stream":false}'

# 9. Gateway — parse (sync JSON)
curl -X POST http://localhost:8003/api/ai/gateway \
  -H "Content-Type: application/json" \
  -d '{"task_type":"parse","text":"John Doe, 25, lives at 123 Main St.","stream":false}'

# 10. Gateway — agent chat
curl -X POST http://localhost:8003/api/ai/gateway \
  -H "Content-Type: application/json" \
  -d '{"task_type":"agent","text":"What is the capital of France?","stream":false}'

# 11. Gateway — async process
curl -X POST http://localhost:8003/api/ai/gateway \
  -H "Content-Type: application/json" \
  -d '{"task_type":"process","text":"Process this in the background."}'

# 12. Check job status (use job_id from step 11)
curl http://localhost:8003/api/ai/status/<job_id>
```

### Hub Compatibility Endpoints

```bash
# 13. Chat stream (hub format)
curl -X POST http://localhost:8003/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"user_id":"test"}'

# 14. Generate embedding
curl -X POST http://localhost:8003/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Embed this text","user_id":"test"}'

# 15. Extract text from a file
curl -X POST http://localhost:8003/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"file_path":"/path/to/file.txt","file_type":"txt","user_id":"test"}'

# 16. RAG: ingest chunks
curl -X POST http://localhost:8003/api/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","document_id":"doc1","text":"Some content to store in the vector database."}'

# 17. RAG: retrieve chunks
curl -X POST http://localhost:8003/api/v1/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query":"some content","top_k":3}'

# 18. RAG: delete document
curl -X DELETE "http://localhost:8003/api/v1/rag/documents/doc1?user_id=test"
```

If `SMARTHUB_API_KEY` is set in `.env`, add `-H "X-API-KEY: your-key"` to all requests targeting `/api/ai/*` and `/api/v1/*` endpoints.

## Troubleshooting

**"API key is a placeholder" warning at startup:**
→ Edit `.env` with real API keys from your provider

**"Tesseract system binary may be missing" on image extraction:**
→ macOS: `brew install tesseract`  
→ Linux: `apt install tesseract-ocr`

**"Address already in use" on port 8003:**
→ `lsof -ti:8003 | xargs kill -9`

**All LLM calls fail with 500:**
→ Check your API keys in `.env`  
→ For Ollama: ensure `ollama serve` is running
