"""Сценарий использования для индексации документов по путям файловой системы."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from datetime import datetime
from time import time
from typing import Iterable
from uuid import uuid4

from domain.entities import Chunk, Document, EmbeddingSpec
from domain.interfaces import ChunkRepository, ChunkSplitter, DocumentRepository, Embedder, EmbeddingStore, TextExtractor
from application.use_cases.embedding_utils import get_embedder_for_spec
from infrastructure.text_extraction.docx_extractor import DocxExtractor
from infrastructure.text_extraction.html_extractor import HtmlExtractor
from infrastructure.text_extraction.pdf_extractor import PdfExtractor
from infrastructure.text_extraction.plain_text_extractor import PlainTextExtractor


@dataclass(slots=True)
class IngestError:
    path: str
    reason: str


@dataclass(slots=True)
class IngestReport:
    total: int
    indexed: int
    errors: list[IngestError] = field(default_factory=list)


_EXTENSION_EXTRACTORS: dict[str, TextExtractor] = {
    ".pdf": PdfExtractor(),
    ".docx": DocxExtractor(),
    ".txt": PlainTextExtractor(),
    ".md": PlainTextExtractor(),
    ".html": HtmlExtractor(),
    ".htm": HtmlExtractor(),
}


def ingest_paths(
    paths: Iterable[Path],
    *,
    splitter: ChunkSplitter,
    embedders: dict[str, Embedder],
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
    chunk_repository: ChunkRepository,
    embedding_specs: list[EmbeddingSpec],
) -> IngestReport:
    """Индексировать документы по списку файлов или директорий."""

    files = _collect_files(paths)
    report = IngestReport(total=len(files), indexed=0)

    for path in files:
        indexed_at = int(time())
        metadata = _build_metadata(path, indexed_at=indexed_at)
        try:
            extractor = _EXTENSION_EXTRACTORS[path.suffix.lower()]
            raw_bytes = path.read_bytes()
            text = extractor.extract(raw_bytes)
            metadata["content_hash"] = sha256(raw_bytes).hexdigest()
        except Exception as exc:  # pragma: no cover - защитный код
            metadata["error"] = str(exc)
            document = Document(id=_document_id(path, metadata), content="", metadata=metadata)
            document_repository.add(document)
            report.errors.append(IngestError(path=str(path), reason=str(exc)))
            continue

        document = Document(
            id="",
            path=str(path.resolve()),
            title=path.name,
            mime_type=metadata.get("mime_type"),
            content=text,
            content_hash=metadata.get("content_hash"),
            modified_at=datetime.fromtimestamp(metadata.get("mtime", 0)),
            metadata=metadata,
        )
        document_repository.add(document)
        report.indexed += 1

        chunks = splitter.split(document)
        persisted_chunks: list[Chunk] = []
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata.setdefault("source_uri", metadata["source_uri"])
            chunk.text_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
            if not chunk.id:
                chunk.id = str(uuid4())
            chunk_repository.add(chunk)
            persisted_chunks.append(chunk)
        if not chunks:
            continue
        _index_embeddings(
            embedders=embedders,
            embedding_store=embedding_store,
            embedding_specs=embedding_specs,
            document=document,
            chunks=persisted_chunks,
        )

    return report


def _collect_files(paths: Iterable[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in _EXTENSION_EXTRACTORS:
                    collected.append(file_path)
        elif path.is_file():
            if path.suffix.lower() in _EXTENSION_EXTRACTORS:
                collected.append(path)
    return collected


def _build_metadata(path: Path, *, indexed_at: int) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_type": "file",
        "source_uri": str(path.resolve()),
        "display_name": path.name,
        "extension": path.suffix.lower().lstrip("."),
        "mime_type": _guess_mime(path),
        "size_bytes": stat.st_size,
        "mtime": int(stat.st_mtime),
        "content_hash": "",
        "indexed_at": indexed_at,
    }


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext in {".html", ".htm"}:
        return "text/html"
    return "text/plain"


def _index_embeddings(
    *,
    embedders: dict[str, Embedder],
    embedding_store: EmbeddingStore,
    embedding_specs: list[EmbeddingSpec],
    document: Document,
    chunks: list[Chunk],
) -> None:
    for spec in embedding_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        if spec.level == "document":
            doc_embedding = embedder.embed_texts([document.content])[0]
            embedding_store.add(spec, "document", [document.id], [doc_embedding])
        else:
            embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
            embedding_store.add(spec, "chunk", [chunk.id for chunk in chunks], embeddings)


__all__ = ["ingest_paths", "IngestReport", "IngestError"]
