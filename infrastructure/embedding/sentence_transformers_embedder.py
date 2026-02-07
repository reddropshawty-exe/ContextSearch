"""Эмбеддеры на базе sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import logging

from sentence_transformers import SentenceTransformer

from domain.entities import Query
from domain.interfaces import Embedder


@dataclass(slots=True)
class SentenceTransformersConfig:
    model_name: str
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 16
    query_prefix: str | None = None
    passage_prefix: str | None = None


logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(Embedder):
    """Эмбеддер на базе библиотеки sentence-transformers."""

    def __init__(self, config: SentenceTransformersConfig) -> None:
        self._config = config
        logger.info("Загрузка модели sentence-transformers: %s", config.model_name)
        self._model = SentenceTransformer(config.model_name, device=config.device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _apply_prefix(self, text: str, prefix: str | None) -> str:
        if prefix:
            return f"{prefix}{text}"
        return text

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self._apply_prefix(text, self._config.passage_prefix) for text in texts]
        logger.debug("Кодирование %d документов моделью %s", len(prefixed), self._config.model_name)
        embeddings = self._model.encode(
            prefixed,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: Query) -> list[float]:
        text = self._apply_prefix(query.text, self._config.query_prefix)
        logger.debug("Кодирование запроса моделью %s", self._config.model_name)
        embeddings = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self._config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings[0].tolist()


__all__ = ["SentenceTransformersEmbedder", "SentenceTransformersConfig"]
