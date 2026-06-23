from app.services.vector_store import chunk_text


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_only_whitespace(self):
        assert chunk_text("   ") == []

    def test_shorter_than_chunk_size(self):
        result = chunk_text("hello world", chunk_size=100)
        assert result == ["hello world"]

    def test_longer_than_chunk_size(self):
        text = "word " * 20
        result = chunk_text(text.strip(), chunk_size=10, overlap=0)
        assert len(result) == 2

    def test_exact_chunk_size(self):
        text = " ".join(str(i) for i in range(10))
        result = chunk_text(text, chunk_size=10, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_with_overlap(self):
        text = "word " * 15
        result = chunk_text(text.strip(), chunk_size=10, overlap=5)
        assert len(result) >= 2

    def test_overlap_capped_at_chunk_minus_one(self):
        text = "a b c d e f g h i j"
        result = chunk_text(text, chunk_size=5, overlap=10)
        assert len(result) >= 1

    def test_negative_overlap_clamped(self):
        text = "a b c d e f g h i j"
        result = chunk_text(text, chunk_size=5, overlap=-5)
        assert len(result) == 2

    def test_chunk_size_zero(self):
        assert chunk_text("hello world", chunk_size=0) == []

    def test_chunk_size_negative(self):
        assert chunk_text("hello world", chunk_size=-1) == []

    def test_chunk_size_one(self):
        text = "a b c"
        result = chunk_text(text, chunk_size=1, overlap=0)
        assert result == ["a", "b", "c"]

    def test_single_word(self):
        assert chunk_text("hello", chunk_size=500) == ["hello"]
