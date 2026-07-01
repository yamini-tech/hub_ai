import os
import tempfile
import time
from unittest.mock import patch

import pytest
import yaml

from app.services.config_watcher import ConfigWatcher

SAMPLE_CONFIG = {
    "model_list": [
        {
            "model_name": "smart-hub-summarizer",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "os.environ/OPENAI_API_KEY"},
        },
        {
            "model_name": "smart-hub-parser",
            "litellm_params": {"model": "anthropic/claude-3-5-sonnet", "api_key": "os.environ/ANTHROPIC_API_KEY"},
        },
        {
            "model_name": "smart-hub-reasoner",
            "litellm_params": {"model": "anthropic/claude-3-7-sonnet", "api_key": "os.environ/ANTHROPIC_API_KEY"},
        },
    ],
    "fallback_chains": {
        "gpt-4o": ["gpt-4o-mini", "ollama/llama3.2"],
        "claude-3-5-sonnet": ["claude-3-7-sonnet", "gpt-4o-mini", "ollama/llama3.2"],
        "claude-3-7-sonnet": ["gpt-4o-mini", "ollama/llama3.2"],
        "ollama/llama3.2": [],
    },
}


@pytest.fixture
def config_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(SAMPLE_CONFIG, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestConfigWatcher:
    def test_loads_model_map(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_model("smart-hub-summarizer") == "openai/gpt-4o"
        assert watcher.get_model("smart-hub-parser") == "anthropic/claude-3-5-sonnet"

    def test_loads_fallback_chains(self, config_file):
        watcher = ConfigWatcher(config_file)
        fallback = watcher.get_fallback("gpt-4o")
        assert "gpt-4o-mini" in fallback
        assert "ollama/llama3.2" in fallback

    def test_loads_task_model_map(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_task_model("summarize") == "openai/gpt-4o"
        assert watcher.get_task_model("parse") == "anthropic/claude-3-5-sonnet"

    def test_unknown_model_returns_none(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_model("nonexistent") is None

    def test_unknown_task_returns_none(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_task_model("nonexistent") is None

    def test_no_fallback_for_unknown_model(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_fallback("unknown/model") == []

    def test_fallback_by_prefix(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert "gpt-4o-mini" in watcher.get_fallback("gpt-4o-preview")

    def test_fallback_by_suffix(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert "ollama/llama3.2" in watcher.get_fallback("some/gpt-4o")

    def test_reload_detects_change(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_model("smart-hub-summarizer") == "openai/gpt-4o"

        # Wait to ensure mtime changes
        time.sleep(0.01)
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "model_list": [
                        {"model_name": "smart-hub-summarizer", "litellm_params": {"model": "ollama/llama3.2"}},
                    ]
                },
                f,
            )

        reloaded = watcher.reload()
        assert reloaded is True
        assert watcher.get_model("smart-hub-summarizer") == "ollama/llama3.2"

    def test_reload_no_change_returns_false(self, config_file):
        watcher = ConfigWatcher(config_file)
        reloaded = watcher.reload()
        assert reloaded is False

    def test_force_reload(self, config_file):
        watcher = ConfigWatcher(config_file)
        assert watcher.get_model("smart-hub-summarizer") == "openai/gpt-4o"

        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "model_list": [
                        {"model_name": "smart-hub-summarizer", "litellm_params": {"model": "claude-3-5-sonnet"}},
                    ]
                },
                f,
            )

        watcher.reload_force()
        assert watcher.get_model("smart-hub-summarizer") == "claude-3-5-sonnet"

    def test_default_model_fallback(self, config_file):
        with patch.dict(os.environ, {"SMARTHUB_MODEL": "custom-default"}):
            watcher = ConfigWatcher(config_file)
            assert "smart-hub-parser" in watcher.get_model_map()
            assert watcher.get_default_model() == "custom-default"

    def test_empty_config_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            f.flush()
            fname = f.name

        try:
            watcher = ConfigWatcher(fname)
            assert watcher.get_model("smart-hub-summarizer") is None
            # Falls back to default model when no config entry matches
            assert watcher.get_task_model("parse") is not None
            # Fallback chains should still have defaults
            assert watcher.get_fallback("gpt-4o")
        finally:
            os.unlink(fname)

    def test_get_model_map_returns_copy(self, config_file):
        watcher = ConfigWatcher(config_file)
        m1 = watcher.get_model_map()
        m2 = watcher.get_model_map()
        m1["injected"] = "test"
        assert "injected" not in m2

    def test_get_fallback_chains_returns_copy(self, config_file):
        watcher = ConfigWatcher(config_file)
        f1 = watcher.get_fallback_chains()
        f2 = watcher.get_fallback_chains()
        f1["injected"] = []
        assert "injected" not in f2
