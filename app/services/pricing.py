PRICING_TABLE = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-7-sonnet": {"prompt": 3.00, "completion": 15.00},
    "llama3-8b": {"prompt": 0.05, "completion": 0.10},
    "llama3.2": {"prompt": 0.00, "completion": 0.00},
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_TABLE.get(model_name, {"prompt": 0.5, "completion": 1.5})
    prompt_cost = (prompt_tokens / 1_000_000) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * rates["completion"]
    return round(prompt_cost + completion_cost, 6)
