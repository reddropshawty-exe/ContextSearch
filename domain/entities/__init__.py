"""Domain entities for the ContextSearch system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Represents a raw document that can be ingested."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A chunk of a larger document used for retrieval."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    """A user query issued to the system."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Result returned after running retrieval for a query."""

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
