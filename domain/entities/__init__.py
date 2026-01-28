"""Доменные сущности системы ContextSearch."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Document:
    """Представляет сырой документ, который можно индексировать."""

    id: str
    path: str | None = None
    title: str | None = None
    mime_type: str | None = None
    content: str = ""
    content_hash: str | None = None
    modified_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Фрагмент документа, используемый для поиска."""

    id: str
    document_id: str
    text: str
    start: int | None = None
    end: int | None = None
    text_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    """Пользовательский поисковый запрос."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Результат поиска по запросу."""

    document_id: str
    score: float
    chunk: Chunk | None = None
    chunk_text_preview: str | None = None
    document: Document | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingSpec:
    """Спецификация пространства эмбеддингов."""

    id: str
    model_name: str
    dimension: int
    metric: str
    normalize: bool
    level: str
    params: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Document",
    "Chunk",
    "Query",
    "RetrievalResult",
    "EmbeddingSpec",
]
