"""SQLite-репозиторий для маппинга ann_id к объектам."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from domain.interfaces import EmbeddingRecordRepository


class SqliteEmbeddingRecordRepository(EmbeddingRecordRepository):
    """Хранит маппинг ann_id к объектам в SQLite."""

    def __init__(self, db_path: str | Path = "contextsearch.db") -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_records (
                    embedding_spec_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    ann_id INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(embedding_spec_id, ann_id),
                    UNIQUE(embedding_spec_id, object_type, object_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_records_spec_ann
                ON embedding_records (embedding_spec_id, ann_id)
                """
            )

    def add_records(
        self,
        spec_id: str,
        object_type: str,
        object_ids: Sequence[str],
        ann_ids: Sequence[int],
    ) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_records (
                    embedding_spec_id, object_type, object_id, ann_id
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (spec_id, object_type, object_id, ann_id)
                    for object_id, ann_id in zip(object_ids, ann_ids)
                ],
            )

    def find_object_ids(self, spec_id: str, ann_ids: Sequence[int]) -> list[str]:
        if not ann_ids:
            return []
        placeholders = ",".join("?" for _ in ann_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ann_id, object_id FROM embedding_records
                WHERE embedding_spec_id = ? AND ann_id IN ({placeholders})
                """,
                (spec_id, *ann_ids),
            ).fetchall()
        ann_to_object = {row[0]: row[1] for row in rows}
        return [ann_to_object[ann_id] for ann_id in ann_ids if ann_id in ann_to_object]

    def list_object_ids(self, spec_id: str, object_type: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT object_id FROM embedding_records
                WHERE embedding_spec_id = ? AND object_type = ?
                """,
                (spec_id, object_type),
            ).fetchall()
        return [row[0] for row in rows]


__all__ = ["SqliteEmbeddingRecordRepository"]
