from unittest.mock import MagicMock, patch

import pytest

from app.services.model_selector import select_model


@pytest.fixture
def mock_watcher():
    """Fixture that patches get_config_watcher to return a controlled mock."""
    watcher = MagicMock()
    watcher.get_default_model.return_value = "ollama/llama3.2"
    watcher.get_task_model.side_effect = lambda t: {
        "summarize": "openai/gpt-4o",
        "chat": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
        "parse": "anthropic/claude-3-5-sonnet",
        "agent": "claude-3-5-sonnet",
    }.get(t)
    watcher.get_model.side_effect = lambda n: {
        "smart-hub-reasoner": "anthropic/claude-3-7-sonnet",
    }.get(n)
    watcher.get_fallback.return_value = []

    with patch("app.services.model_selector.get_config_watcher", return_value=watcher):
        yield watcher


class TestSelectModel:
    def test_known_task_type(self, mock_watcher):
        model = select_model("summarize")
        assert model == "openai/gpt-4o"

    def test_unknown_task_type_falls_back(self, mock_watcher):
        mock_watcher.get_task_model.return_value = None
        model = select_model("nonexistent")
        assert model == "ollama/llama3.2"

    @patch("app.services.model_selector._TOKEN_THRESHOLD", 5)
    def test_chat_long_text_uses_reasoner(self, mock_watcher):
        model = select_model("chat", text="this is a long text that exceeds the threshold")
        assert model == "anthropic/claude-3-7-sonnet"

    @patch("app.services.model_selector._TOKEN_THRESHOLD", 100)
    def test_chat_short_text_uses_task_model(self, mock_watcher):
        model = select_model("chat", text="short")
        assert model == "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"

    @patch("app.services.model_selector._TOKEN_THRESHOLD", 1)
    def test_no_reasoner_in_map_falls_back(self, mock_watcher):
        def get_model_side_effect(name):
            return None  # No reasoner for any model name

        mock_watcher.get_model.side_effect = get_model_side_effect
        model = select_model("chat", text="long text here")
        assert model == "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"

    @patch("app.services.model_selector._TOKEN_THRESHOLD", 5)
    def test_agent_long_text_uses_reasoner(self, mock_watcher):
        model = select_model("agent", text="this is a long agent text")
        assert model == "anthropic/claude-3-7-sonnet"
