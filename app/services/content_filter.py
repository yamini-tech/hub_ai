import re

_FILTER_PATTERNS = [
    (re.compile(r"\b(kill|murder|hurt|harm|attack|torture)\s+(yourself|everyone|people|them|someone)\b"),
     "Content blocked: violent content"),
    (re.compile(r"\b(how\s+to\s+)?(make|build|create|synthesize|manufacture)\s+(a\s+)?(bomb|explosive|weapon|gun|poison|drug|narcotic)\b"),
     "Content blocked: harmful instructions"),
    (re.compile(r"\b(self[\s-]?(harm|hurt|kill|destruct)|suicide|cutting)\b"),
     "Content blocked: self-harm content"),
    (re.compile(r"\b(fuck|shit|asshole|bitch|cunt|motherfucker)\b"),
     "Content blocked: profanity"),
    (re.compile(r"\b(nigger|faggot|kike|spic|chink|raghead|gook|cracker|white\s*trash)\b"),
     "Content blocked: hate speech"),
    (re.compile(r"\b(child\s*(porn|abuse|exploit)|cp\s*(content|link)|minor\s*abuse)\b"),
     "Content blocked: illegal content"),
    (re.compile(r"\b(how\s+to\s+)?(rape|molest|assault|abuse)\s+(a\s+)?(child|minor|woman|someone)\b"),
     "Content blocked: violent content"),
    (re.compile(r"\bexplicit\s+(sexual|adult|nude|porn|xxx)\b"),
     "Content blocked: explicit content"),
    (re.compile(r"\b(sql\s*inject|drop\s+table|rm\s+-rf|\/dev\/random|xss\s+attack|cross.site.scripting)\s*\(?\)?"),
     "Content blocked: malicious code"),
]

_BLOCKED_RESPONSE = "I'm unable to generate that response. Please rephrase your request."


def check_content(text: str) -> tuple[bool, str]:
    if not isinstance(text, str) or not text.strip():
        return True, ""
    lower = text.lower()
    for pattern, reason in _FILTER_PATTERNS:
        if pattern.search(lower):
            return False, reason
    return True, ""


async def filter_stream(stream):
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            safe, _ = check_content(content)
            if not safe:
                chunk.choices[0].delta.content = "[CONTENT FILTERED]"
        yield chunk
