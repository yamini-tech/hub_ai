import os

PRICING_TABLE = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-7-sonnet": {"prompt": 3.00, "completion": 15.00},
    "llama3-8b": {"prompt": 0.05, "completion": 0.10},
    "llama3.2": {"prompt": 0.00, "completion": 0.00},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.00},
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-7-sonnet-20250219": {"prompt": 3.00, "completion": 15.00},
    "llama3-8b-8192": {"prompt": 0.05, "completion": 0.10},
}


def _normalize_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    name = _normalize_model_name(model_name)
    rates = PRICING_TABLE.get(name, {"prompt": 0.5, "completion": 1.5})
    prompt_cost = (prompt_tokens / 1_000_000) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * rates["completion"]
    return round(prompt_cost + completion_cost, 6)
