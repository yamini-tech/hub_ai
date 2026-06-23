import pytest
from unittest.mock import patch
from app.services.model_selector import select_model


MODEL_NAME = "ollama/llama3.2"


class TestSelectModel:
    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {
        "summarize": "openai/gpt-4o",
        "chat": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
        "parse": "anthropic/claude-3-5-sonnet",
    })
    def test_known_task_type(self):
        model = select_model("summarize")
        assert model == "openai/gpt-4o"

    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {})
    def test_unknown_task_type_falls_back(self):
        model = select_model("nonexistent")
        assert model == MODEL_NAME

    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {
        "chat": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    })
    @patch("app.services.model_selector.MODEL_MAP", {
        "smart-hub-reasoner": "anthropic/claude-3-7-sonnet",
    })
    @patch("app.services.model_selector._TOKEN_THRESHOLD", 5)
    def test_chat_long_text_uses_reasoner(self):
        model = select_model("chat", text="this is a long text that exceeds the threshold")
        assert model == "anthropic/claude-3-7-sonnet"

    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {
        "chat": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    })
    @patch("app.services.model_selector.MODEL_MAP", {
        "smart-hub-reasoner": "anthropic/claude-3-7-sonnet",
    })
    @patch("app.services.model_selector._TOKEN_THRESHOLD", 100)
    def test_chat_short_text_uses_task_model(self):
        model = select_model("chat", text="short")
        assert model == "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"

    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {
        "chat": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    })
    @patch("app.services.model_selector.MODEL_MAP", {})
    @patch("app.services.model_selector._TOKEN_THRESHOLD", 1)
    def test_no_reasoner_in_map_falls_back(self):
        model = select_model("chat", text="long text here")
        assert model == "huggingface/meta-llama/Meta-Llama-3-8B-Instruct"

    @patch("app.services.model_selector.MODEL_NAME", MODEL_NAME)
    @patch("app.services.model_selector.TASK_MODEL_MAP", {
        "agent": "claude-3-5-sonnet",
    })
    @patch("app.services.model_selector.MODEL_MAP", {
        "smart-hub-reasoner": "claude-3-7-sonnet",
    })
    @patch("app.services.model_selector._TOKEN_THRESHOLD", 5)
    def test_agent_long_text_uses_reasoner(self):
        model = select_model("agent", text="this is a long agent text")
        assert model == "claude-3-7-sonnet"
