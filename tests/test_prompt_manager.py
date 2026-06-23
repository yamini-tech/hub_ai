import os
import tempfile
import yaml
import pytest
from app.services.prompt_manager import _PROMPTS, load_prompts, get_system_prompt


@pytest.fixture(autouse=True)
def clear_prompts():
    _PROMPTS.clear()
    yield
    _PROMPTS.clear()


class TestLoadPrompts:
    def test_load_from_path(self):
        data = {
            "system_prompts": {
                "default": "Default prompt",
                "test_task": "Test task prompt",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name

        try:
            load_prompts(path)
            assert _PROMPTS["default"] == "Default prompt"
            assert _PROMPTS["test_task"] == "Test task prompt"
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        load_prompts("/nonexistent/path.yaml")
        assert _PROMPTS == {}

    def test_load_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: : : :")
            path = f.name

        try:
            load_prompts(path)
            assert _PROMPTS == {}
        finally:
            os.unlink(path)

    def test_load_no_system_prompts_key(self):
        data = {"other_key": {"something": "value"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name

        try:
            load_prompts(path)
            assert _PROMPTS == {}
        finally:
            os.unlink(path)


class TestGetSystemPrompt:
    def test_get_known_task(self):
        _PROMPTS["summarize"] = "Summarize this text."
        result = get_system_prompt("summarize")
        assert result == "Summarize this text."

    def test_unknown_task_falls_back_to_default(self):
        _PROMPTS["default"] = "Default prompt."
        result = get_system_prompt("nonexistent_task")
        assert result == "Default prompt."

    def test_unknown_task_no_default_returns_empty(self):
        result = get_system_prompt("nonexistent_task")
        assert result == ""

    def test_with_kwargs_formatting(self):
        _PROMPTS["agent"] = "Context: {context}"
        result = get_system_prompt("agent", context="my context")
        assert result == "Context: my context"

    def test_without_kwargs_when_template_has_placeholder(self):
        _PROMPTS["agent"] = "Context: {context}"
        result = get_system_prompt("agent")
        assert result == "Context: {context}"
