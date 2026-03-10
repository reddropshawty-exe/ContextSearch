"""Персистентное хранилище эмбеддингов в SQLite."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

from domain.entities import EmbeddingSpec
from domain.interfaces import EmbeddingRecordRepository, EmbeddingStore


class SqliteEmbeddingStore(EmbeddingStore):
    """Хранит все эмбеддинги в SQLite и ищет перебором."""

    def __init__(
        self,
        *,
        db_path: str | Path = "contextsearch.db",
        record_repository: EmbeddingRecordRepository,
    ) -> None:
        self._db_path = Path(db_path)
        self._record_repository = record_repository
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_vectors (
                    embedding_spec_id TEXT NOT NULL,
                    ann_id INTEGER NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (embedding_spec_id, ann_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_vectors_spec_object
                ON embedding_vectors (embedding_spec_id, object_type, object_id)
                """
            )

    def add(
        self,
        spec: EmbeddingSpec,
        object_type: str,
        object_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not object_ids:
            return
        ann_ids = self._allocate_ids(spec.id, len(object_ids))
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_vectors (
                    embedding_spec_id, ann_id, object_type, object_id, vector
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (spec.id, ann_id, object_type, object_id, json.dumps(list(vector)))
                    for ann_id, object_id, vector in zip(ann_ids, object_ids, embeddings)
                ],
            )
        self._record_repository.add_records(spec.id, object_type, object_ids, ann_ids)

    def search(
        self,
        spec: EmbeddingSpec,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ann_id, vector
                FROM embedding_vectors
                WHERE embedding_spec_id = ?
                """,
                (spec.id,),
            ).fetchall()
        if not rows:
            return []

        scored: list[tuple[int, float]] = []
        for ann_id, raw_vec in rows:
            vector = json.loads(raw_vec)
            score = self._cosine_similarity(query_embedding, vector)
            scored.append((ann_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _allocate_ids(self, spec_id: str, count: int) -> list[int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ann_id), -1)
                FROM embedding_vectors
                WHERE embedding_spec_id = ?
                """,
                (spec_id,),
            ).fetchone()
        start = int(row[0]) + 1 if row is not None else 0
        return list(range(start, start + count))

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b))
        denom_a = sum(x * x for x in a) ** 0.5 or 1.0
        denom_b = sum(x * x for x in b) ** 0.5 or 1.0
        return numerator / (denom_a * denom_b)


__all__ = ["SqliteEmbeddingStore"]
