import json
from typing import Any, Optional

try:
    import jsonschema as _jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    _jsonschema = None
    HAS_JSONSCHEMA = False


def validate_json_output(content: str, schema: Optional[dict] = None) -> tuple[bool, str, Any]:
    """Validate LLM output as valid JSON and optionally against a JSON Schema.
    
    Args:
        content: Raw string output from the LLM.
        schema: Optional JSON Schema dict to validate against.
        
    Returns:
        (is_valid, error_message, parsed_data)
        - is_valid: True if JSON is valid and schema passes (if provided)
        - error_message: empty if valid, description if invalid
        - parsed_data: parsed Python object if valid, None if JSON is malformed
    """
    if not content or not content.strip():
        return False, "Empty response — expected valid JSON", None

    if content.startswith("```"):
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("```"):
                content = "\n".join(lines[i+1:]).strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
                break

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None

    if schema is not None and HAS_JSONSCHEMA:
        try:
            _jsonschema.validate(data, schema)
        except _jsonschema.ValidationError as e:
            return False, f"Schema validation failed: {e.message}", data

    return True, "", data
