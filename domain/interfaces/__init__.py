"""Abstract interfaces for the ContextSearch system."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from domain.entities import Chunk, Document, Query, RetrievalResult


class TextExtractor(ABC):
    """Extracts text from user provided sources (files, URLs, etc.)."""

    @abstractmethod
    def extract(self, source: bytes | str) -> str:
        """Return the textual representation of a source."""


class ChunkSplitter(ABC):
    """Splits documents into semantic chunks for retrieval."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Return chunks for the provided document."""


class Embedder(ABC):
    """Turns text (documents or queries) into vector embeddings."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the stable identifier for this embedding model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an iterable of texts into dense vectors."""

    @abstractmethod
    def embed_query(self, query: Query) -> list[float]:
        """Embed a user query for retrieval."""


class EmbeddingStore(ABC):
    """Persists embeddings and provides similarity search."""

    collection_id: str

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Store embeddings for the provided chunks."""

    @abstractmethod
    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[RetrievalResult]:
        """Return the best matching chunks for the provided query embedding."""


class DocumentRepository(ABC):
    """Persists metadata for documents."""

    @abstractmethod
    def add(self, document: Document) -> None:
        """Store a document record."""

    @abstractmethod
    def list(self) -> list[Document]:
        """Return all stored documents."""

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        """Retrieve a document by id."""


class ChunkRepository(ABC):
    """Persists chunk metadata and content."""

    @abstractmethod
    def add(self, chunk: Chunk) -> int:
        """Store a chunk and return its integer id."""

    @abstractmethod
    def get(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk by id."""


class QueryRewriter(ABC):
    """Allows experimentation with query rewriting strategies."""

    @abstractmethod
    def rewrite(self, query: Query) -> list[Query]:
        """Return rewritten variants for the provided query."""


class Reranker(ABC):
    """Applies a reranking strategy after initial retrieval."""

    @abstractmethod
    def rerank(self, query: Query, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
        """Return a reranked list of retrieval results."""


__all__ = [
    "TextExtractor",
    "ChunkSplitter",
    "Embedder",
    "EmbeddingStore",
    "DocumentRepository",
    "ChunkRepository",
    "QueryRewriter",
    "Reranker",
]
