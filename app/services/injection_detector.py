import re

_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?(previous\s+)?(instructions|directives|commands|rules)"),
     "Prompt injection detected: override attempt"),
    (re.compile(r"(you\s+are\s+(now|free|no\s+longer)\s+|pretend\s+(to\s+)?be|act\s+as\s+if\s+you)"),
     "Prompt injection detected: role-play override"),
    (re.compile(r"(system\s+(prompt|message|instruction)|new\s+instructions?)"),
     "Prompt injection detected: system prompt override"),
    (re.compile(r"\bdan\b|do\s+anything\s+now|jail\s*break"),
     "Prompt injection detected: jailbreak attempt"),
    (re.compile(r"output\s+(in\s+)?base64|encoded\s+in\s+base64"),
     "Prompt injection detected: encoding bypass"),
    (re.compile(r"(forget|ignore|discard)\s+(everything|all|your|previous)"),
     "Prompt injection detected: instruction override"),
    (re.compile(r"you\s+(have\s+been|are\s+)\s*hacked"),
     "Prompt injection detected: system compromise claim"),
    (re.compile(r"(print|show|reveal|output|display|leak)\s+(your\s+)?(system\s+)?(prompt|instructions?|message)"),
     "Prompt injection detected: prompt extraction"),
    (re.compile(r"\[\s*system\s*\]|<\|im_start\|>|<\|im_end\|>"),
     "Prompt injection detected: token injection"),
    (re.compile(r"forget\s+(the\s+)?previous\s+(prompt|instructions|context)"),
     "Prompt injection detected: context reset"),
]

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def check_injection(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    normalized = _normalize(text)
    for pattern, message in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return message
    return None
