import os

import yaml

from app.services.config_watcher import ConfigWatcher, get_config_watcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Default model config — overridden by config.yaml if present
MODEL_NAME = os.getenv("SMARTHUB_MODEL", "ollama/llama3.2")
API_BASE = os.getenv("SMARTHUB_API_BASE", "http://localhost:11434")

# ── Initial load from config.yaml ────────────────────────────────────
# These module-level globals are populated once at import time for
# backward compatibility. The ConfigWatcher singleton provides live
# hot-reload access used by services that need it.

MODEL_MAP: dict[str, str] = {}
FALLBACK_CHAINS: dict[str, list[str]] = {
    "gpt-4o": ["gpt-4o-mini", "ollama/llama3.2"],
    "gpt-4o-mini": ["ollama/llama3.2"],
    "claude-3-5-sonnet": ["claude-3-7-sonnet", "gpt-4o-mini", "ollama/llama3.2"],
    "claude-3-7-sonnet": ["gpt-4o-mini", "ollama/llama3.2"],
    "ollama/llama3.2": [],
}

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        if cfg and "model_list" in cfg:
            for entry in cfg["model_list"]:
                name = entry.get("model_name", "")
                params = entry.get("litellm_params", {})
                model = params.get("model", "")
                if name and model:
                    MODEL_MAP[name] = model
        if cfg and "fallback_chains" in cfg and cfg["fallback_chains"]:
            FALLBACK_CHAINS = cfg["fallback_chains"]
except Exception as e:
    print(f"WARNING: Failed to load config/config.yaml: {e}. Falling back to default models.")

TASK_MODEL_MAP: dict[str, str] = {
    "summarize": MODEL_MAP.get("smart-hub-summarizer", MODEL_NAME),
    "chat": MODEL_MAP.get("smart-hub-hf-chat", MODEL_NAME),
    "extraction": MODEL_MAP.get("smart-hub-parser", MODEL_NAME),
    "reasoning": MODEL_MAP.get("smart-hub-reasoner", MODEL_NAME),
    "parse": MODEL_MAP.get("smart-hub-parser", MODEL_NAME),
}
TEST_MODEL = os.getenv("SMARTHUB_TEST_MODEL", "")
if TEST_MODEL:
    TASK_MODEL_MAP = {k: TEST_MODEL for k in TASK_MODEL_MAP.keys()}

# ── Hot-reload support ───────────────────────────────────────────────


def reload_config() -> dict:
    """Force a reload of config.yaml and return status info.

    Thread-safe. All services using the ConfigWatcher singleton pick up
    the new config on their next call.
    """
    watcher = get_config_watcher()
    watcher.reload_force()
    # Also update module-level vars for any code that still imports them
    _sync_globals(watcher)
    return {"status": "reloaded"}


def poll_config() -> bool:
    """Check for config changes and reload if needed. Returns True if reloaded."""
    watcher = get_config_watcher()
    if watcher.reload():
        _sync_globals(watcher)
        print("INFO: config.yaml changed — model routing reloaded.")
        return True
    return False


def _sync_globals(watcher: ConfigWatcher) -> None:
    """Sync module-level globals from the watcher (for code not using watcher directly)."""
    global MODEL_MAP, FALLBACK_CHAINS, TASK_MODEL_MAP
    MODEL_MAP = watcher.get_model_map()
    FALLBACK_CHAINS = watcher.get_fallback_chains()
    TASK_MODEL_MAP = {
        "summarize": watcher.get_model("smart-hub-summarizer") or MODEL_NAME,
        "chat": watcher.get_model("smart-hub-hf-chat") or MODEL_NAME,
        "extraction": watcher.get_model("smart-hub-parser") or MODEL_NAME,
        "reasoning": watcher.get_model("smart-hub-reasoner") or MODEL_NAME,
        "parse": watcher.get_model("smart-hub-parser") or MODEL_NAME,
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

# Token Limits
MAX_TOKENS_GATEWAY = int(os.getenv("SMARTHUB_MAX_TOKENS_GATEWAY", "10000"))
MAX_TOKENS_TASK = int(os.getenv("SMARTHUB_MAX_TOKENS_TASK", "20000"))
