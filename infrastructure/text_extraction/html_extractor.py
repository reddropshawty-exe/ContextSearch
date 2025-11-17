"""HTML extractor that strips tags and normalises whitespace."""
from __future__ import annotations

from html.parser import HTMLParser

from domain.interfaces import TextExtractor


class _CollectingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - trivial
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        text = " ".join(self._parts)
        self._parts.clear()
        return text


class HtmlExtractor(TextExtractor):
    """Very small HTML extractor using Python's built-in parser."""

    def extract(self, source: bytes | str) -> str:
        if isinstance(source, bytes):
            raw = source.decode("utf-8", errors="ignore")
        else:
            raw = source
        parser = _CollectingParser()
        parser.feed(raw)
        return parser.get_text()


__all__ = ["HtmlExtractor"]
