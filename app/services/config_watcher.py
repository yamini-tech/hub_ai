import os
import threading

import yaml

CONFIG_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "config",
    "config.yaml",
)

POLL_INTERVAL_SEC = 30


class ConfigWatcher:
    """Loads config.yaml and supports hot-reload on file change.

    Thread-safe: all public methods acquire a read lock.
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self._config_path = config_path
        self._lock = threading.RLock()
        self._model_map: dict[str, str] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._task_model_map: dict[str, str] = {}
        self._last_mtime: float = 0
        self._default_model: str = os.getenv("SMARTHUB_MODEL", "ollama/llama3.2")
        self._load()

    # ── loading ──────────────────────────────────────────────────────

    def _load(self) -> None:
        model_map: dict[str, str] = {}
        fallback_chains: dict[str, list[str]] = {
            "gpt-4o": ["gpt-4o-mini", "ollama/llama3.2"],
            "gpt-4o-mini": ["ollama/llama3.2"],
            "claude-3-5-sonnet": ["claude-3-7-sonnet", "gpt-4o-mini", "ollama/llama3.2"],
            "claude-3-7-sonnet": ["gpt-4o-mini", "ollama/llama3.2"],
            "ollama/llama3.2": [],
        }

        try:
            if os.path.exists(self._config_path):
                with open(self._config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg and "model_list" in cfg:
                    for entry in cfg["model_list"]:
                        name = entry.get("model_name", "")
                        params = entry.get("litellm_params", {})
                        model = params.get("model", "")
                        if name and model:
                            model_map[name] = model
                if cfg and "fallback_chains" in cfg and cfg["fallback_chains"]:
                    fallback_chains = cfg["fallback_chains"]
                self._last_mtime = os.path.getmtime(self._config_path)
        except Exception as e:
            print(f"WARNING: Failed to load config.yaml: {e}. Keeping previous config.")

        self._model_map = model_map
        self._fallback_chains = fallback_chains
        self._task_model_map = {
            "summarize": model_map.get("smart-hub-summarizer", self._default_model),
            "chat": model_map.get("smart-hub-hf-chat", self._default_model),
            "extraction": model_map.get("smart-hub-parser", self._default_model),
            "reasoning": model_map.get("smart-hub-reasoner", self._default_model),
            "parse": model_map.get("smart-hub-parser", self._default_model),
        }

    # ── public API ───────────────────────────────────────────────────

    def reload(self) -> bool:
        """Reload config if the file has changed since last load.

        Returns True if a reload occurred, False otherwise.
        """
        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            return False

        if mtime <= self._last_mtime:
            return False

        with self._lock:
            self._load()
            return True

    def reload_force(self) -> None:
        """Force a reload regardless of mtime."""
        with self._lock:
            self._load()

    def get_model(self, name: str) -> str | None:
        with self._lock:
            return self._model_map.get(name)

    def get_fallback(self, model: str) -> list[str]:
        with self._lock:
            for prefix, fallbacks in self._fallback_chains.items():
                if model.startswith(prefix) or model.endswith(prefix):
                    return fallbacks
            return []

    def get_task_model(self, task_type: str) -> str | None:
        with self._lock:
            return self._task_model_map.get(task_type)

    def get_default_model(self) -> str:
        return self._default_model

    def get_model_map(self) -> dict[str, str]:
        with self._lock:
            return dict(self._model_map)

    def get_fallback_chains(self) -> dict[str, list[str]]:
        with self._lock:
            return dict(self._fallback_chains)


# Module-level singleton
_watcher: ConfigWatcher | None = None
_watcher_lock = threading.Lock()


def get_config_watcher() -> ConfigWatcher:
    global _watcher
    if _watcher is None:
        with _watcher_lock:
            if _watcher is None:
                _watcher = ConfigWatcher()
    return _watcher
