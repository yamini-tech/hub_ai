from app.services.json_validator import validate_json_output


class TestValidateJsonOutput:
    def test_valid_json_object(self):
        content = '{"name": "John", "age": 30}'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert error == ""
        assert data == {"name": "John", "age": 30}

    def test_valid_json_array(self):
        content = "[1, 2, 3]"
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == [1, 2, 3]

    def test_invalid_json(self):
        content = '{"name": "John"'
        valid, error, data = validate_json_output(content)
        assert valid is False
        assert "Invalid JSON" in error
        assert data is None

    def test_empty_content(self):
        valid, error, data = validate_json_output("")
        assert valid is False
        assert "Empty response" in error
        assert data is None

    def test_only_whitespace(self):
        valid, error, data = validate_json_output("   ")
        assert valid is False
        assert "Empty response" in error

    def test_strips_markdown_fences(self):
        content = '```json\n{"key": "value"}\n```'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == {"key": "value"}

    def test_strips_markdown_fences_without_lang(self):
        content = '```\n{"key": "value"}\n```'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == {"key": "value"}

    def test_nested_json(self):
        content = '{"user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}}}'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data["user"]["address"]["city"] == "NYC"

    def test_boolean_and_null(self):
        content = '{"flag": true, "value": null, "items": []}'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == {"flag": True, "value": None, "items": []}

    def test_valid_against_schema(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        content = '{"name": "John", "age": 30}'
        valid, error, data = validate_json_output(content, schema=schema)
        assert valid is True
        assert error == ""

    def test_invalid_against_schema(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        content = '{"name": "John", "age": "thirty"}'
        valid, error, data = validate_json_output(content, schema=schema)
        assert valid is False
        assert "Schema validation failed" in error
        assert data is not None

    def test_schema_missing_required_field(self):
        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        content = '{"name": "John"}'
        valid, error, data = validate_json_output(content, schema=schema)
        assert valid is False
        assert "Schema validation failed" in error

    def test_schema_with_array_items(self):
        schema = {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        content = '{"items": ["a", "b", "c"]}'
        valid, error, data = validate_json_output(content, schema=schema)
        assert valid is True

    def test_schema_array_invalid_item(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        }
        content = '{"items": [1, "two", 3]}'
        valid, error, data = validate_json_output(content, schema=schema)
        assert valid is False
        assert "Schema validation failed" in error

    def test_non_dict_json_passes_without_schema(self):
        content = '"just a string"'
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == "just a string"

    def test_number_json(self):
        content = "42"
        valid, error, data = validate_json_output(content)
        assert valid is True
        assert data == 42
