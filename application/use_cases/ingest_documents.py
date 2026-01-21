"""Сценарий использования для индексации документов в системе."""
from __future__ import annotations

from hashlib import sha256
from time import time
from typing import Iterable

from domain.entities import Document
from domain.interfaces import (
    ChunkSplitter,
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    TextExtractor,
)


def ingest_documents(
    sources: Iterable[tuple[str | None, bytes | str]],
    *,
    extractor: TextExtractor,
    splitter: ChunkSplitter,
    embedder: Embedder,
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
) -> list[Document]:
    """Индексировать набор источников, идентифицированных id."""

    ingested_documents: list[Document] = []
    for document_id, source in sources:
        indexed_at = int(time())
        text = extractor.extract(source)
        if document_id is None:
            document_id = _fallback_document_id(text, indexed_at)
        document = Document(
            id=document_id,
            content=text,
            metadata={
                "source_type": "raw",
                "source_uri": document_id,
                "display_name": document_id,
                "extension": "",
                "size_bytes": len(text.encode("utf-8")),
                "mtime": indexed_at,
                "content_hash": sha256(text.encode("utf-8")).hexdigest(),
                "indexed_at": indexed_at,
            },
        )
        document_repository.add(document)
        ingested_documents.append(document)

        chunks = splitter.split(document)
        if not chunks:
            continue
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata.setdefault("collection_id", embedding_store.collection_id)
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        embedding_store.add(chunks, embeddings)

    return ingested_documents


def _fallback_document_id(text: str, indexed_at: int) -> str:
    digest = sha256(f"{indexed_at}:{text}".encode("utf-8")).hexdigest()
    return digest
