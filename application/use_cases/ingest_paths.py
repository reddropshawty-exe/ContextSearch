"""Use case for ingesting documents from file system paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Iterable

from domain.entities import Document
from domain.interfaces import ChunkSplitter, DocumentRepository, Embedder, EmbeddingStore, TextExtractor
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
    embedder: Embedder,
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
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

        document = Document(id=_document_id(path, metadata), content=text, metadata=metadata)
        document_repository.add(document)
        report.indexed += 1

        chunks = splitter.split(document)
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata.setdefault("source_uri", metadata["source_uri"])
            chunk.metadata.setdefault("collection_id", embedding_store.collection_id)
        if not chunks:
            continue
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        embedding_store.add(chunks, embeddings)

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
        "size_bytes": stat.st_size,
        "mtime": int(stat.st_mtime),
        "content_hash": "",
        "indexed_at": indexed_at,
    }


def _document_id(path: Path, metadata: dict[str, object]) -> str:
    base = f"{path.resolve()}|{metadata.get('mtime')}|{metadata.get('size_bytes')}"
    return sha256(base.encode("utf-8")).hexdigest()


__all__ = ["ingest_paths", "IngestReport", "IngestError"]
