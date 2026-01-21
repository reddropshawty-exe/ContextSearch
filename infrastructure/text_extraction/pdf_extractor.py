"""PDF-экстрактор на базе pdfplumber."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pdfplumber

from domain.interfaces import TextExtractor


class PdfExtractor(TextExtractor):
    """Извлекает текст из PDF-байтов или путей к файлам."""

    def extract(self, source: bytes | str) -> str:
        if isinstance(source, bytes):
            stream = BytesIO(source)
            with pdfplumber.open(stream) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
        path = Path(source)
        if path.exists():
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
        return source


__all__ = ["PdfExtractor"]
