"""Базовый переписыватель запроса, возвращающий исходный запрос."""
from __future__ import annotations

from domain.entities import Query
from domain.interfaces import QueryRewriter


class SimpleQueryRewriter(QueryRewriter):
    """Возвращает исходный запрос без изменений."""

    def rewrite(self, query: Query) -> list[Query]:  # pragma: no cover - тривиально
        return [query]


__all__ = ["SimpleQueryRewriter"]
