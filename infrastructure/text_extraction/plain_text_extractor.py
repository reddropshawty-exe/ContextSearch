"""Экстрактор текста, воспринимающий вход как UTF-8."""
from __future__ import annotations

from domain.interfaces import TextExtractor


class PlainTextExtractor(TextExtractor):
    """Простой экстрактор для уже очищенного текста."""

    def extract(self, source: bytes | str) -> str:  # pragma: no cover - тривиально
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="ignore")
        return source


__all__ = ["PlainTextExtractor"]
