import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
FILE_PATH = os.path.join(PROJECT_ROOT, "knowledge.txt")

# Default model config — overridden by config.yaml if present
MODEL_NAME = os.getenv("SMARTHUB_MODEL", "ollama/llama3.2")
API_BASE = os.getenv("SMARTHUB_API_BASE", "http://localhost:11434")

# Model routing from config.yaml
MODEL_MAP = {}

config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
try:
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if cfg and "model_list" in cfg:
            for entry in cfg["model_list"]:
                name = entry.get("model_name", "")
                params = entry.get("litellm_params", {})
                model = params.get("model", "")
                if name and model:
                    MODEL_MAP[name] = model
except Exception:
    pass  # fall back to defaults

# Fallback model map for task types
TASK_MODEL_MAP = {
    "summarize": MODEL_MAP.get("smart-hub-summarizer", MODEL_NAME),
    "chat": MODEL_MAP.get("smart-hub-hf-chat", MODEL_NAME),
    "extraction": MODEL_MAP.get("smart-hub-parser", MODEL_NAME),
    "reasoning": MODEL_MAP.get("smart-hub-reasoner", MODEL_NAME),
    "parse": MODEL_MAP.get("smart-hub-parser", MODEL_NAME),
}

# Security
API_KEY = os.getenv("SMARTHUB_API_KEY", "")
API_KEY_ENABLED = bool(API_KEY)

# Warn about placeholder API keys
_placeholder_patterns = ["xxxxx", "your-", "sk-proj-xxxxxxxx"]
for _var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY", "HUGGINGFACE_API_KEY"]:
    _val = os.getenv(_var, "")
    if any(p in _val for p in _placeholder_patterns):
        print(f"WARNING: {_var} in .env is a placeholder — calls to this provider will fail.")

# Throttling
RATE_LIMIT_WINDOW_SEC = int(os.getenv("SMARTHUB_RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("SMARTHUB_RATE_LIMIT_MAX", "30"))
