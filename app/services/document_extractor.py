import os

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_BASE = os.environ.get(
    "SMARTHUB_ALLOWED_EXTRACT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


def _sanitize_path(file_path: str) -> str:
    resolved = os.path.realpath(os.path.abspath(file_path))
    if not resolved.startswith(ALLOWED_BASE):
        raise ValueError(f"Access denied: path outside allowed directory ({resolved})")
    return resolved


def _check_file_size(file_path: str):
    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE} bytes)")


def _get_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext.lower().lstrip(".")


def extract_text(file_path: str, file_type: str) -> str:
    file_type = file_type.lower().lstrip(".")

    if file_type not in ("txt", "pdf", "docx", "png", "jpg", "jpeg"):
        raise ValueError(f"Unsupported file type: {file_type}")

    actual_ext = _get_extension(file_path)
    if actual_ext and actual_ext != file_type:
        raise ValueError(f"File extension '{actual_ext}' does not match declared type '{file_type}'")

    safe_path = _sanitize_path(file_path)
    _check_file_size(safe_path)

    if file_type == "txt":
        return _extract_txt(safe_path)
    elif file_type == "pdf":
        return _extract_pdf(safe_path)
    elif file_type == "docx":
        return _extract_docx(safe_path)
    else:
        return _extract_image(safe_path)

def _extract_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def _extract_pdf(file_path: str) -> str:
    import fitz
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)

def _extract_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_image(file_path: str) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "[Image OCR requires 'pytesseract' package. Install with: pip install pytesseract]"
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "[No text detected in image]"
    except Exception:
        return (
            "[Image OCR failed. Tesseract system binary may be missing.\n"
            "  Install with: brew install tesseract (macOS)\n"
            "  Install with: apt install tesseract-ocr (Linux)]"
        )
