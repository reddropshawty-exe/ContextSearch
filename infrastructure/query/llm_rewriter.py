"""Локальный LLM-переписыватель запросов для экспериментов."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from transformers import pipeline

from domain.entities import Query
from domain.interfaces import QueryRewriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRewriterConfig:
    model: str = "google/flan-t5-small"
    max_queries: int = 3
    max_query_length: int = 256
    max_new_tokens: int = 64


class LLMQueryRewriter(QueryRewriter):
    """Генерирует несколько вариантов запроса с помощью локальной LLM."""

    def __init__(self, config: LLMRewriterConfig) -> None:
        self._config = config
        self._pipeline = pipeline("text2text-generation", model=config.model)

    def rewrite(self, query: Query) -> list[Query]:
        try:
            raw = self._request_rewrites(query.text)
        except Exception:  # pragma: no cover - делаем лучшее возможное
            logger.exception("Не удалось переписать запрос через LLM.")
            return [query]

        candidates = self._parse_candidates(raw)
        if not candidates:
            return [query]
        return [Query(text=text) for text in candidates]

    def _request_rewrites(self, text: str) -> str:
        prompt = (
            "Сгенерируй {count} альтернативных поисковых запросов для запроса ниже. "
            "Верни каждый вариант с новой строки без нумерации.\n\n"
            "Запрос: {query}"
        ).format(count=self._config.max_queries, query=text)
        outputs = self._pipeline(prompt, max_new_tokens=self._config.max_new_tokens, do_sample=False)
        return outputs[0]["generated_text"]

    def _parse_candidates(self, raw: str) -> list[str]:
        cleaned = raw.strip()
        if not cleaned:
            return []
        candidates = cleaned.splitlines()
        normalized: list[str] = []
        for candidate in candidates:
            text = candidate.strip(" -\t\r\n")
            if not text:
                continue
            text = text[: self._config.max_query_length]
            if text not in normalized:
                normalized.append(text)
        return normalized[: self._config.max_queries]


__all__ = ["LLMQueryRewriter", "LLMRewriterConfig"]
