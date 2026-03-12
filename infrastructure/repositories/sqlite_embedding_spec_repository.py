"""SQLite-репозиторий для спецификаций эмбеддингов."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from domain.entities import EmbeddingSpec
from domain.interfaces import EmbeddingSpecRepository


class SqliteEmbeddingSpecRepository(EmbeddingSpecRepository):
    """Хранит спецификации эмбеддингов в SQLite."""

    def __init__(self, db_path: str | Path = "contextsearch.db") -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_specs (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    normalize INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    params TEXT NOT NULL
                )
                """
            )

    def upsert(self, spec: EmbeddingSpec) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                REPLACE INTO embedding_specs (id, model_name, dimension, metric, normalize, level, params)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.id,
                    spec.model_name,
                    spec.dimension,
                    spec.metric,
                    int(spec.normalize),
                    spec.level,
                    json.dumps(spec.params),
                ),
            )

    def list(self) -> list[EmbeddingSpec]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, model_name, dimension, metric, normalize, level, params
                FROM embedding_specs
                """
            ).fetchall()
        specs: list[EmbeddingSpec] = []
        for row in rows:
            specs.append(
                EmbeddingSpec(
                    id=row[0],
                    model_name=row[1],
                    dimension=row[2],
                    metric=row[3],
                    normalize=bool(row[4]),
                    level=row[5],
                    params=json.loads(row[6]),
                )
            )
        return specs

    def get(self, spec_id: str) -> EmbeddingSpec | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, model_name, dimension, metric, normalize, level, params
                FROM embedding_specs WHERE id = ?
                """,
                (spec_id,),
            ).fetchone()
        if row is None:
            return None
        return EmbeddingSpec(
            id=row[0],
            model_name=row[1],
            dimension=row[2],
            metric=row[3],
            normalize=bool(row[4]),
            level=row[5],
            params=json.loads(row[6]),
        )


__all__ = ["SqliteEmbeddingSpecRepository"]
