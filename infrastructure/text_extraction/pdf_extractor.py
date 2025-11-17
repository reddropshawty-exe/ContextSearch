"""Simple PDF extractor placeholder that works with bytes or strings."""
from __future__ import annotations

from domain.interfaces import TextExtractor


class PdfExtractor(TextExtractor):
    """A naive extractor that treats the input as UTF-8 text."""

    def extract(self, source: bytes | str) -> str:  # pragma: no cover - trivial
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="ignore")
        return source


__all__ = ["PdfExtractor"]
