"""Chunk splitter that uses a fixed-size sliding window."""
from __future__ import annotations

import uuid

from domain.entities import Chunk, Document
from domain.interfaces import ChunkSplitter


class FixedWindowSplitter(ChunkSplitter):
    """Split documents using a fixed-size window and stride."""

    def __init__(self, window_size: int = 400, stride: int = 350) -> None:
        self.window_size = window_size
        self.stride = stride

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks: list[Chunk] = []
        source_uri = document.metadata.get("source_uri")
        for start in range(0, len(text), self.stride):
            fragment = text[start : start + self.window_size]
            if not fragment:
                break
            end = min(start + self.window_size, len(text))
            metadata = {"start": start, "end": end}
            if source_uri:
                metadata["source_uri"] = source_uri
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    text=fragment,
                    metadata=metadata,
                )
            )
            if start + self.window_size >= len(text):
                break
        return chunks


__all__ = ["FixedWindowSplitter"]
