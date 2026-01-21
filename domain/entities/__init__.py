"""Доменные сущности системы ContextSearch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Представляет сырой документ, который можно индексировать."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Фрагмент документа, используемый для поиска."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    """Пользовательский поисковый запрос."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Результат поиска по запросу."""

    chunk: Chunk
    score: float
    document: Document | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Document",
    "Chunk",
    "Query",
    "RetrievalResult",
]
