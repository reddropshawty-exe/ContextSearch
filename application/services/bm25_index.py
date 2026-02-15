"""BM25 индекс с кэшированием для повторного использования между запросами."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from domain.entities import Document


@dataclass(slots=True)
class _State:
    index: BM25Okapi | None
    doc_ids: list[str]
    fingerprint: str


class BM25Index:
    """Кэширует BM25 индекс, перестраивая его только при изменении коллекции."""

    def __init__(self) -> None:
        self._state = _State(index=None, doc_ids=[], fingerprint="")

    def update_documents(self, documents: list[Document]) -> None:
        fingerprint = self._fingerprint(documents)
        if fingerprint == self._state.fingerprint:
            return
        corpus = [self._tokenize(doc.content) for doc in documents]
        if not documents or not any(corpus):
            self._state = _State(index=None, doc_ids=[], fingerprint=fingerprint)
            return
        self._state = _State(
            index=BM25Okapi(corpus),
            doc_ids=[doc.id for doc in documents],
            fingerprint=fingerprint,
        )

    def scores(self, query_text: str) -> dict[str, float]:
        if self._state.index is None:
            return {}
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return {}
        values = self._state.index.get_scores(query_tokens)
        return {doc_id: float(score) for doc_id, score in zip(self._state.doc_ids, values)}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _fingerprint(documents: list[Document]) -> str:
        parts = [f"{doc.id}:{doc.content_hash or ''}:{len(doc.content)}" for doc in documents]
        return "|".join(sorted(parts))


__all__ = ["BM25Index"]
