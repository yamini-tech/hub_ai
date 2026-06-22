import os

def extract_text(file_path: str, file_type: str) -> str:
    file_type = file_type.lower().lstrip(".")

    if file_type == "txt":
        return _extract_txt(file_path)
    elif file_type == "pdf":
        return _extract_pdf(file_path)
    elif file_type == "docx":
        return _extract_docx(file_path)
    elif file_type in ("png", "jpg", "jpeg"):
        return _extract_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

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
