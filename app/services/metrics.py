from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "smartbrain_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "smartbrain_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

llm_requests_total = Counter(
    "smartbrain_llm_requests_total",
    "Total LLM API calls",
    ["model", "status"],
)

llm_tokens_total = Counter(
    "smartbrain_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "token_type"],
)
