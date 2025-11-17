"""Basic query rewriter that keeps the original query."""
from __future__ import annotations

from domain.entities import Query
from domain.interfaces import QueryRewriter


class SimpleQueryRewriter(QueryRewriter):
    """Return the original query without modifications."""

    def rewrite(self, query: Query) -> list[Query]:  # pragma: no cover - trivial
        return [query]


__all__ = ["SimpleQueryRewriter"]
