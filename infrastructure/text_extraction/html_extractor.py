"""HTML-экстрактор, удаляющий теги и нормализующий пробелы."""
from __future__ import annotations

from html.parser import HTMLParser

from domain.interfaces import TextExtractor


class _CollectingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:  # pragma: no cover - тривиально
        if self._ignored_depth:
            return
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        text = " ".join(self._parts)
        self._parts.clear()
        return text


class HtmlExtractor(TextExtractor):
    """Минимальный HTML-экстрактор на базе встроенного парсера Python."""

    def extract(self, source: bytes | str) -> str:
        if isinstance(source, bytes):
            raw = source.decode("utf-8", errors="ignore")
        else:
            raw = source
        parser = _CollectingParser()
        parser.feed(raw)
        return parser.get_text()


__all__ = ["HtmlExtractor"]
