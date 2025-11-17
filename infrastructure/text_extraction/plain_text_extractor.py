"""Text extractor that treats the payload as plain UTF-8 text."""
from __future__ import annotations

from domain.interfaces import TextExtractor


class PlainTextExtractor(TextExtractor):
    """Simple extractor for already-clean text blobs."""

    def extract(self, source: bytes | str) -> str:  # pragma: no cover - trivial
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="ignore")
        return source


__all__ = ["PlainTextExtractor"]
