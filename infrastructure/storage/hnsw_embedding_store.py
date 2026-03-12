"""ANN-хранилище на базе hnswlib."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import hnswlib
import numpy as np

from domain.entities import EmbeddingSpec
from domain.interfaces import EmbeddingRecordRepository, EmbeddingStore

logger = logging.getLogger(__name__)


class HnswEmbeddingStore(EmbeddingStore):
    """Хранит индексы hnswlib для каждого EmbeddingSpec."""

    def __init__(
        self,
        *,
        index_root: str | Path = "indexes",
        record_repository: EmbeddingRecordRepository,
    ) -> None:
        self._index_root = Path(index_root)
        self._index_root.mkdir(parents=True, exist_ok=True)
        self._record_repository = record_repository
        self._indexes: dict[str, hnswlib.Index] = {}
        self._next_ids: dict[str, int] = {}

    def add(
        self,
        spec: EmbeddingSpec,
        object_type: str,
        object_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not object_ids:
            return
        index = self._get_or_create_index(spec)
        vectors = np.array(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != spec.dimension:
            raise ValueError(
                "Неверная размерность эмбеддингов для спецификации "
                f"{spec.id}: ожидалась {spec.dimension}, получено {vectors.shape}."
            )
        if spec.normalize:
            vectors = self._normalize(vectors)

        ann_ids = self._allocate_ids(spec.id, len(object_ids))
        index.add_items(vectors, ann_ids)
        self._record_repository.add_records(spec.id, object_type, object_ids, ann_ids)
        self._save_index(spec)

    def search(
        self,
        spec: EmbeddingSpec,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        index = self._get_or_create_index(spec)
        if index.get_current_count() == 0:
            return []
        index.set_ef(max(spec.params.get("ef_search", 50), top_k))
        vector = np.array([query_embedding], dtype="float32")
        if spec.normalize:
            vector = self._normalize(vector)
        ann_ids, distances = index.knn_query(vector, k=top_k)
        scores = self._scores_from_distances(spec.metric, distances[0])
        return list(zip(ann_ids[0].tolist(), scores))

    def _get_or_create_index(self, spec: EmbeddingSpec) -> hnswlib.Index:
        index = self._indexes.get(spec.id)
        if index is not None:
            return index
        index_path = self._index_path(spec.id)
        index = hnswlib.Index(space=spec.metric, dim=spec.dimension)
        if index_path.exists():
            logger.info("Загрузка HNSW индекса %s", index_path)
            index.load_index(str(index_path))
            self._next_ids[spec.id] = index.get_current_count()
        else:
            index.init_index(
                max_elements=100_000,
                ef_construction=spec.params.get("ef_construction", 200),
                M=spec.params.get("M", 16),
            )
            index.set_ef(spec.params.get("ef_search", 50))
            self._next_ids[spec.id] = 0
        self._indexes[spec.id] = index
        return index

    def _save_index(self, spec: EmbeddingSpec) -> None:
        index = self._indexes.get(spec.id)
        if index is None:
            return
        index_path = self._index_path(spec.id)
        index.save_index(str(index_path))
        meta_path = self._meta_path(spec.id)
        meta_path.write_text(json.dumps({"next_id": self._next_ids.get(spec.id, 0)}, indent=2), encoding="utf-8")

    def _index_path(self, spec_id: str) -> Path:
        return self._index_root / f"{spec_id}.bin"

    def _meta_path(self, spec_id: str) -> Path:
        return self._index_root / f"{spec_id}.json"

    def _allocate_ids(self, spec_id: str, count: int) -> list[int]:
        start = self._next_ids.get(spec_id, 0)
        ann_ids = list(range(start, start + count))
        self._next_ids[spec_id] = start + count
        return ann_ids

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    @staticmethod
    def _scores_from_distances(metric: str, distances: Sequence[float]) -> list[float]:
        if metric == "cosine":
            return [1.0 - float(distance) for distance in distances]
        if metric == "ip":
            return [float(distance) for distance in distances]
        return [float(distance) for distance in distances]


__all__ = ["HnswEmbeddingStore"]
