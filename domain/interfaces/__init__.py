"""Абстрактные интерфейсы системы ContextSearch."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from domain.entities import Chunk, Document, Query, RetrievalResult


class TextExtractor(ABC):
    """Извлекает текст из источников пользователя (файлы, URL и т.д.)."""

    @abstractmethod
    def extract(self, source: bytes | str) -> str:
        """Вернуть текстовое представление источника."""


class ChunkSplitter(ABC):
    """Делит документы на чанки для поиска."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Вернуть чанки для заданного документа."""


class Embedder(ABC):
    """Преобразует текст (документы или запросы) в векторные эмбеддинги."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Вернуть стабильный идентификатор модели эмбеддингов."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Вернуть размерность эмбеддингов модели."""

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
        """Преобразовать список текстов в плотные вектора."""

    @abstractmethod
    def embed_query(self, query: Query) -> list[float]:
        """Преобразовать запрос пользователя в вектор для поиска."""


class EmbeddingStore(ABC):
    """Хранит эмбеддинги и выполняет поиск по сходству."""

    collection_id: str

    collection_id: str

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Сохранить эмбеддинги для указанных чанков."""

    @abstractmethod
    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[RetrievalResult]:
        """Вернуть лучшие совпадения для эмбеддинга запроса."""


class DocumentRepository(ABC):
    """Хранит метаданные документов."""

    @abstractmethod
    def add(self, document: Document) -> None:
        """Сохранить документ."""

    @abstractmethod
    def list(self) -> list[Document]:
        """Вернуть список всех документов."""

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        """Получить документ по идентификатору."""


class ChunkRepository(ABC):
    """Хранит метаданные и содержимое чанков."""

    @abstractmethod
    def add(self, chunk: Chunk) -> int:
        """Сохранить чанк и вернуть его числовой идентификатор."""

    @abstractmethod
    def get(self, chunk_id: int) -> Chunk | None:
        """Получить чанк по идентификатору."""


class ChunkRepository(ABC):
    """Persists chunk metadata and content."""

    @abstractmethod
    def add(self, chunk: Chunk) -> int:
        """Store a chunk and return its integer id."""

    @abstractmethod
    def get(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk by id."""


class QueryRewriter(ABC):
    """Позволяет экспериментировать со стратегиями переписывания запросов."""

    @abstractmethod
    def rewrite(self, query: Query) -> list[Query]:
        """Вернуть варианты переписанного запроса."""


class Reranker(ABC):
    """Применяет стратегию переранжирования после первичного поиска."""

    @abstractmethod
    def rerank(self, query: Query, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
        """Вернуть список результатов после переранжирования."""


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
