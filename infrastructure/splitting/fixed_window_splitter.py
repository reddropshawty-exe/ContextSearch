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
        for start in range(0, len(text), self.stride):
            fragment = text[start : start + self.window_size]
            if not fragment:
                break
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    text=fragment,
                    metadata={"start": start},
                )
            )
            if start + self.window_size >= len(text):
                break
        return chunks


__all__ = ["FixedWindowSplitter"]
