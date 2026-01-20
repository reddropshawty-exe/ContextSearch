"""SQLite-backed repository for chunks."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from domain.entities import Chunk
from domain.interfaces import ChunkRepository


class SqliteChunkRepository(ChunkRepository):
    """Stores chunks inside the shared SQLite database."""

    def __init__(self, db_path: str | Path = "contextsearch.db") -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)")

    def add(self, chunk: Chunk) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chunks (document_id, text, metadata) VALUES (?, ?, ?)",
                (chunk.document_id, chunk.text, json.dumps(chunk.metadata)),
            )
            return int(cursor.lastrowid)

    def get(self, chunk_id: int) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_id, document_id, text, metadata FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk(id=str(row[0]), document_id=row[1], text=row[2], metadata=json.loads(row[3]))


__all__ = ["SqliteChunkRepository"]
