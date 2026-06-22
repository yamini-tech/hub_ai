import litellm
from fastapi import HTTPException
from app.core.config import MODEL_NAME

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge_base",
            "description": "Read the saved research notes from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }
]

async def call_llm(messages: list, tools_list=None, response_format=None, model=None):
    try:
        response = await litellm.acompletion(
            model=model or MODEL_NAME,
            messages=messages, tools=tools_list, response_format=response_format
        )
        return response.choices[0].message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_llm_stream(messages: list, tools_list=None, model=None):
    try:
        response = await litellm.acompletion(
            model=model or MODEL_NAME,
            messages=messages, tools=tools_list,
            stream=True
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
