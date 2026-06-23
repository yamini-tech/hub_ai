from app.services.pii_redactor import redact_pii


class TestPiiRedactor:
    def test_redacts_email(self):
        result = redact_pii("contact me at user@example.com please")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_redacts_phone(self):
        result = redact_pii("call +1-555-123-4567 now")
        assert "[PHONE]" in result

    def test_redacts_ssn(self):
        result = redact_pii("SSN: 123-45-6789")
        assert "[SSN]" in result

    def test_redacts_credit_card(self):
        result = redact_pii("card 4111-1111-1111-1111")
        assert "[CC]" in result

    def test_no_pii_unchanged(self):
        result = redact_pii("hello world this is safe")
        assert result == "hello world this is safe"

    def test_non_string_returns_as_is(self):
        assert redact_pii(123) == 123
        assert redact_pii(None) is None
