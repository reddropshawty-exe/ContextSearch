"""SQLite-репозиторий для чанков."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from domain.entities import Chunk
from domain.interfaces import ChunkRepository


class SqliteChunkRepository(ChunkRepository):
    """Хранит чанки в общей SQLite-базе."""

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
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start INTEGER,
                    end INTEGER,
                    text_hash TEXT,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)")
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        required = {
            "start": "INTEGER",
            "end": "INTEGER",
            "text_hash": "TEXT",
        }
        for name, column_type in required.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {column_type}")

    def add(self, chunk: Chunk) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                REPLACE INTO chunks (chunk_id, document_id, text, start, end, text_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.text,
                    chunk.start,
                    chunk.end,
                    chunk.text_hash,
                    json.dumps(chunk.metadata),
                ),
            )
            return chunk.id

    def list(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, document_id, text, start, end, text_hash, metadata FROM chunks"
            ).fetchall()
        return [
            Chunk(
                id=row[0],
                document_id=row[1],
                text=row[2],
                start=row[3],
                end=row[4],
                text_hash=row[5],
                metadata=json.loads(row[6]),
            )
            for row in rows
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_id, document_id, text, start, end, text_hash, metadata FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=row[0],
            document_id=row[1],
            text=row[2],
            start=row[3],
            end=row[4],
            text_hash=row[5],
            metadata=json.loads(row[6]),
        )


__all__ = ["SqliteChunkRepository"]
