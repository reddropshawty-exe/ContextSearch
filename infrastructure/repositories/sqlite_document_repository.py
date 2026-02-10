"""SQLite-репозиторий для хранения метаданных документов."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
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
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 1)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    title TEXT,
                    mime_type TEXT,
                    content TEXT,
                    content_hash TEXT,
                    modified_at TEXT,
                    metadata TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        required = {
            "path": "TEXT",
            "title": "TEXT",
            "mime_type": "TEXT",
            "content_hash": "TEXT",
            "modified_at": "TEXT",
        }
        for name, column_type in required.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {column_type}")

    def add(self, document: Document) -> None:
        if not document.id:
            document.id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                REPLACE INTO documents (
                    id, path, title, mime_type, content, content_hash, modified_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.path,
                    document.title,
                    document.mime_type,
                    document.content,
                    document.content_hash,
                    document.modified_at.isoformat() if document.modified_at else None,
                    json.dumps(document.metadata),
                ),
            )

    def list(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, path, title, mime_type, content, content_hash, modified_at, metadata
                FROM documents
                """
            ).fetchall()
        documents: list[Document] = []
        for row in rows:
            modified_at = datetime.fromisoformat(row[6]) if row[6] else None
            documents.append(
                Document(
                    id=row[0],
                    path=row[1],
                    title=row[2],
                    mime_type=row[3],
                    content=row[4] or "",
                    content_hash=row[5],
                    modified_at=modified_at,
                    metadata=json.loads(row[7]),
                )
            )
        return documents

    def get(self, document_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, path, title, mime_type, content, content_hash, modified_at, metadata
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        modified_at = datetime.fromisoformat(row[6]) if row[6] else None
        return Document(
            id=row[0],
            path=row[1],
            title=row[2],
            mime_type=row[3],
            content=row[4] or "",
            content_hash=row[5],
            modified_at=modified_at,
            metadata=json.loads(row[7]),
        )


__all__ = ["SqliteDocumentRepository"]
