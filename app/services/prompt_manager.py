import os
import yaml

_PROMPTS: dict[str, str] = {}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_default_path = os.path.join(PROJECT_ROOT, "config", "system_prompts.yaml")

def load_prompts(path: str | None = None):
    global _PROMPTS
    path = path or _default_path
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
            if data and "system_prompts" in data:
                _PROMPTS.update(data["system_prompts"])
    except Exception:
        pass

def get_system_prompt(task_type: str, **kwargs) -> str:
    template = _PROMPTS.get(task_type) or _PROMPTS.get("default", "")
    if kwargs:
        return template.format(**kwargs)
    return template
