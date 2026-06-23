from app.services.pricing import calculate_cost


class TestCalculateCost:
    def test_known_model(self):
        cost = calculate_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == round(expected, 6)

    def test_unknown_model_uses_default_rates(self):
        cost = calculate_cost("unknown-model", 1000, 500)
        expected = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert cost == round(expected, 6)

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4o-mini", 0, 0)
        assert cost == 0.0

    def test_gpt4o_pricing(self):
        cost = calculate_cost("gpt-4o", 1_000_000, 0)
        assert cost == 2.50

    def test_claude_pricing(self):
        cost = calculate_cost("claude-3-5-sonnet", 0, 1_000_000)
        assert cost == 15.00

    def test_llama_local_no_cost(self):
        cost = calculate_cost("llama3.2", 1_000_000, 1_000_000)
        assert cost == 0.00

    def test_rounding_precision(self):
        cost = calculate_cost("gpt-4o-mini", 1, 1)
        assert isinstance(cost, float)
        assert cost >= 0.0
