"""Shared session history management for chat and gateway routers.

Provides two layers of memory:
1. **chat_histories** — full message list per session (used by tool-call loop within a request).
2. **session_facts** — short factual summaries extracted from Q&A pairs,
   retrieved by keyword overlap and injected into the system prompt.
"""

import re

chat_histories: dict[str, list[dict]] = {}
session_facts: dict[str, list[str]] = {}

MAX_HISTORY_PER_SESSION = 50
MAX_SESSIONS = 1000
MAX_WINDOW_TURNS = 6  # last 6 messages (~3 full turns) kept in the rolling window
_MAX_FACTS = 50  # max stored facts per session


# ── session history (for tool-call loop) ─────────────────────────────


def get_or_create_session(session_id: str, system_prompt: str) -> list[dict]:
    """Return the message list for *session_id*, creating it if needed."""
    if session_id not in chat_histories:
        _evict_sessions()
        chat_histories[session_id] = [{"role": "system", "content": system_prompt}]
    chat_histories[session_id][0]["content"] = system_prompt
    return chat_histories[session_id]


def append_user_message(session_id: str, text: str) -> list[dict]:
    """Append a user message and trim; return the messages list."""
    messages = chat_histories[session_id]
    messages.append({"role": "user", "content": text})
    _trim_session(session_id)
    return messages


def append_assistant_message(session_id: str, content: str) -> None:
    """Append the assistant's final answer and trim."""
    messages = chat_histories[session_id]
    messages.append({"role": "assistant", "content": content})
    _trim_session(session_id)


# ── sliding window (for LLM prompt) ─────────────────────────────────


def get_window_messages(session_id: str) -> list[dict]:
    """Return the last *MAX_WINDOW_TURNS* messages (excluding system)."""
    messages = chat_histories.get(session_id, [])
    non_system = [m for m in messages if m.get("role") != "system"]
    return non_system[-MAX_WINDOW_TURNS:]


# ── session facts (for system prompt injection) ──────────────────────


def store_session_fact(session_id: str, user_text: str, answer: str) -> None:
    """Extract a short factual summary from a Q&A pair and store it."""
    fact = _extract_fact(user_text, answer)
    if not fact:
        return
    facts = session_facts.setdefault(session_id, [])
    # avoid duplicates
    if fact not in facts:
        facts.append(fact)
        if len(facts) > _MAX_FACTS:
            session_facts[session_id] = facts[-_MAX_FACTS:]


def get_relevant_context(session_id: str, current_question: str) -> str:
    """Return stored facts that are topically related to *current_question*.

    For summary/synthesis questions (e.g. "what are we discussing about?"),
    returns ALL stored facts.  Otherwise uses keyword overlap filtering.
    """
    facts = session_facts.get(session_id, [])
    if not facts:
        return ""

    if _is_summary_question(current_question):
        return _format_facts(facts)

    q_tokens = _tokenize(current_question)
    if not q_tokens:
        return ""

    matches = []
    for fact in facts:
        fact_tokens = _tokenize(fact)
        if q_tokens & fact_tokens:
            matches.append(fact)

    return _format_facts(matches) if matches else ""


def _format_facts(facts: list[str]) -> str:
    """Format stored facts as a numbered list for the system prompt."""
    return "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))


def _is_summary_question(text: str) -> bool:
    """Detect questions asking for a summary/synthesis of the conversation."""
    return bool(_SUMMARY_RE.search(text))


_SUMMARY_RE = re.compile(
    r"\b(what|which|how|can\s+you|tell\s+me|please)\b.{0,40}"
    r"\b(discuss|talk|cover|converse|chat|go\s+over|review|summarize|summarise|recap|about|topic|subject|we|our)\b",
    re.IGNORECASE,
)


def _extract_fact(user_text: str, answer: str) -> str:
    """Best-effort extraction of a factual summary from a Q&A pair."""
    text = user_text.strip()
    lower = text.lower()

    # direct declarations: "my name is X", "i like X", "i prefer X"
    m = re.search(r"(?:my|i)\s+(?:name is|name's|like|prefer|love|use|work with)\s+(.{2,60})", lower)
    if m:
        return m.group(0).strip().capitalize()

    # statements of fact: "X is Y", "the Y is Z"
    m = re.search(r"(.{2,40})\s+is\s+(.{2,60})", text)
    if m:
        return m.group(0).strip()

    # fallback: first sentence of user text, truncated
    first = text.split(".")[0].split(",")[0][:80]
    return first if len(first) > 5 else ""


def _tokenize(text: str) -> set[str]:
    """Lowercase word set, stripped of short stopwords."""
    _stop = frozenset(
        {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "this",
            "that",
            "it",
            "its",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "and",
            "or",
            "but",
            "not",
            "no",
            "so",
            "if",
            "then",
            "than",
            "i",
            "me",
            "you",
            "he",
            "she",
            "we",
            "they",
            "them",
            "us",
            "just",
            "also",
            "very",
            "really",
            "only",
            "even",
            "still",
        }
    )
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    return {w for w in words if w not in _stop}


# ── message builder (used by chat and gateway routers) ───────────────


def build_llm_messages(session_id: str, system_prompt: str, user_text: str) -> list[dict]:
    """Build a messages list for the LLM: system + window history + current question.

    Does NOT include the full conversation history — only the rolling window
    and relevant session facts (injected into the system prompt).
    """
    session_ctx = get_relevant_context(session_id, user_text)
    if session_ctx:
        full_prompt = system_prompt.replace("{session_context}", session_ctx)
    else:
        full_prompt = system_prompt.replace("{session_context}", "")
    window = get_window_messages(session_id)
    return [{"role": "system", "content": full_prompt}] + window + [{"role": "user", "content": user_text}]


# ── internal helpers ─────────────────────────────────────────────────


def _trim_session(session_id: str) -> None:
    messages = chat_histories.get(session_id)
    if not messages:
        return
    if len(messages) > MAX_HISTORY_PER_SESSION:
        keep = MAX_HISTORY_PER_SESSION // 2
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        chat_histories[session_id] = system_msgs + other_msgs[-keep:]


def _evict_sessions() -> None:
    if len(chat_histories) > MAX_SESSIONS:
        excess = len(chat_histories) - MAX_SESSIONS
        for key in list(chat_histories.keys())[:excess]:
            del chat_histories[key]
