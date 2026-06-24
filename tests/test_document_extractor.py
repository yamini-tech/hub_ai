import os
import tempfile
import pytest
from app.services.document_extractor import extract_text, _extract_txt


class TestExtractTxt:
    def test_reads_text_content(self):
        content = "Hello, this is a test file."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = _extract_txt(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_reads_multiline_content(self):
        content = "Line 1\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = _extract_txt(path)
            assert result == content
        finally:
            os.unlink(path)

    def test_handles_binary_chars(self):
        content = "Text with \x00 null byte"
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            f.write(content.encode("utf-8", errors="replace"))
            path = f.name

        try:
            result = _extract_txt(path)
            assert "null" in result
        finally:
            os.unlink(path)


class TestExtractText:
    def test_txt_file_type(self):
        content = "Text content"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = extract_text(path, "txt")
            assert result == content
        finally:
            os.unlink(path)

    def test_txt_with_dot_prefix(self):
        content = "Content"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = extract_text(path, ".txt")
            assert result == content
        finally:
            os.unlink(path)

    def test_unsupported_file_type(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("/fake/path", "exe")

    def test_unsupported_file_type_uppercase(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("/fake/path", "EXE")

    def test_extension_mismatch_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            path = f.name
        try:
            with pytest.raises(ValueError, match="does not match declared type"):
                extract_text(path, "pdf")
        finally:
            os.unlink(path)

    def test_extension_match_passes(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            path = f.name
        try:
            result = extract_text(path, "txt")
            assert result == "content"
        finally:
            os.unlink(path)

    def test_file_not_found_txt(self):
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.txt", "txt")

    @pytest.mark.skip(reason="Requires PyMuPDF")
    def test_pdf_extraction(self):
        pass

    @pytest.mark.skip(reason="Requires python-docx")
    def test_docx_extraction(self):
        pass
