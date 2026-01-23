"""SQLite-репозиторий для хранения метаданных документов."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from domain.entities import Document
from domain.interfaces import DocumentRepository


class SqliteDocumentRepository(DocumentRepository):
    """Хранит документы в лёгкой SQLite-базе."""

    def __init__(self, db_path: str | Path = "contextsearch.db") -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT NOT NULL
                )
                """
            )

    def add(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                "REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                (document.id, document.content, json.dumps(document.metadata)),
            )

    def list(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, content, metadata FROM documents").fetchall()
        return [Document(id=row[0], content=row[1], metadata=json.loads(row[2])) for row in rows]

    def get(self, document_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, content, metadata FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(id=row[0], content=row[1], metadata=json.loads(row[2]))


__all__ = ["SqliteDocumentRepository"]
