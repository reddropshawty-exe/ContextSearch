"""Lightweight embedder that simulates MiniLM embeddings."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from domain.entities import Query
from domain.interfaces import Embedder


class MiniLMEmbedder(Embedder):
    """Deterministic hash-based embedder useful for demos/tests."""

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension
        self._model_id = f"hash-minilm-{dimension}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash(text) for text in texts]

    def embed_query(self, query: Query) -> list[float]:
        return self._hash(query.text)


__all__ = ["MiniLMEmbedder"]
