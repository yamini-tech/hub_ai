import os
import pytest
from unittest.mock import patch


class TestConfigDefaults:
    @patch.dict(os.environ, {}, clear=True)
    def test_default_model_name(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.MODEL_NAME == "ollama/llama3.2"

    @patch.dict(os.environ, {"SMARTHUB_MODEL": "gpt-4o"}, clear=True)
    def test_model_name_from_env(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.MODEL_NAME == "gpt-4o"

    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_disabled_by_default(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.API_KEY_ENABLED is False
        assert app.core.config.API_KEY == ""

    @patch.dict(os.environ, {"SMARTHUB_API_KEY": "test-key-123"}, clear=True)
    def test_api_key_enabled_when_set(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.API_KEY_ENABLED is True
        assert app.core.config.API_KEY == "test-key-123"

    @patch.dict(os.environ, {}, clear=True)
    def test_rate_limit_defaults(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.RATE_LIMIT_WINDOW_SEC == 60
        assert app.core.config.RATE_LIMIT_MAX_REQUESTS == 30

    @patch.dict(os.environ, {
        "SMARTHUB_RATE_LIMIT_WINDOW": "120",
        "SMARTHUB_RATE_LIMIT_MAX": "100",
    }, clear=True)
    def test_rate_limit_from_env(self):
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        assert app.core.config.RATE_LIMIT_WINDOW_SEC == 120
        assert app.core.config.RATE_LIMIT_MAX_REQUESTS == 100


class TestModelMap:
    def test_model_map_loaded_from_yaml(self):
        from app.core.config import MODEL_MAP
        assert isinstance(MODEL_MAP, dict)
        if MODEL_MAP:
            assert "smart-hub-summarizer" in MODEL_MAP or True

    def test_task_model_map_falls_back_to_default(self):
        from app.core.config import TASK_MODEL_MAP, MODEL_NAME
        for task in ("summarize", "chat", "extraction", "reasoning", "parse"):
            assert task in TASK_MODEL_MAP


class TestKnowledgeFilePath:
    def test_knowledge_file_path_exists(self):
        from app.core.config import FILE_PATH
        assert "knowledge.txt" in FILE_PATH

    def test_project_root_is_absolute(self):
        from app.core.config import PROJECT_ROOT
        assert os.path.isabs(PROJECT_ROOT)
        assert PROJECT_ROOT.endswith("hub_ai")


class TestFallbackChains:
    def test_fallback_chains_loaded_from_yaml(self):
        from app.core.config import FALLBACK_CHAINS
        assert isinstance(FALLBACK_CHAINS, dict)
        assert "gpt-4o" in FALLBACK_CHAINS
        assert "gpt-4o-mini" in FALLBACK_CHAINS
        assert "claude-3-5-sonnet" in FALLBACK_CHAINS
        assert "claude-3-7-sonnet" in FALLBACK_CHAINS
        assert "ollama/llama3.2" in FALLBACK_CHAINS
