"""LLM-powered query rewriter for experiments."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import requests

from domain.entities import Query
from domain.interfaces import QueryRewriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRewriterConfig:
    provider: str = "ollama"
    model: str = "llama3.1"
    max_queries: int = 3
    max_query_length: int = 256
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_url: str = "https://api.openai.com/v1/chat/completions"


class LLMQueryRewriter(QueryRewriter):
    """Generate multiple query variants using an LLM provider."""

    def __init__(self, config: LLMRewriterConfig) -> None:
        self._config = config

    def rewrite(self, query: Query) -> list[Query]:
        try:
            raw = self._request_rewrites(query.text)
        except Exception:  # pragma: no cover - best effort
            logger.exception("LLM query rewriting failed.")
            return [query]

        candidates = self._parse_candidates(raw)
        if not candidates:
            return [query]
        return [Query(text=text) for text in candidates]

    def _request_rewrites(self, text: str) -> str:
        prompt = (
            "Generate {count} alternative search queries for the user query below. "
            "Return each variant on a new line without numbering.\n\n"
            "Query: {query}"
        ).format(count=self._config.max_queries, query=text)
        if self._config.provider == "openai":
            return self._call_openai(prompt)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self._config.ollama_url}/api/generate",
            json={"model": self._config.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "")

    def _call_openai(self, prompt: str) -> str:
        api_key = self._config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key.")
        response = requests.post(
            self._config.openai_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": "You rewrite search queries."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _parse_candidates(self, raw: str) -> list[str]:
        cleaned = raw.strip()
        if not cleaned:
            return []
        if cleaned.startswith("["):
            try:
                data = json.loads(cleaned)
                candidates = [str(item) for item in data]
            except json.JSONDecodeError:
                candidates = cleaned.splitlines()
        else:
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
