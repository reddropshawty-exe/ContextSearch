"""Локальный LLM-переписыватель запросов для экспериментов."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.entities import Query
from domain.interfaces import QueryRewriter

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = "Перепиши поисковый запрос в соответствии с инструкцией. Верни только новый запрос без пояснений."


@dataclass(slots=True)
class LLMRewriterConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_query_length: int = 256
    max_new_tokens: int = 64


class LLMQueryRewriter(QueryRewriter):
    """Генерирует один переписанный вариант запроса с помощью локальной LLM."""

    def __init__(self, config: LLMRewriterConfig) -> None:
        self._config = config
        self._tokenizer = None
        self._model = None

    def rewrite(self, query: Query) -> list[Query]:
        if not self._ensure_model():
            return [query]
        try:
            raw = self._request_rewrite(query.text)
        except Exception:  # pragma: no cover - делаем лучшее возможное
            logger.exception("Не удалось переписать запрос через LLM.")
            return [query]

        candidate = self._parse_candidate(raw)
        logger.info("LLM rewrite: model=%s, source=%r, rewritten=%r", self._config.model, query.text, candidate)
        if not candidate:
            return [query]
        return [Query(text=candidate)]

    def _ensure_model(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._config.model)
            self._model = AutoModelForCausalLM.from_pretrained(self._config.model)
            return True
        except Exception:
            logger.exception("Не удалось загрузить LLM rewriter model: %s", self._config.model)
            return False

    def _request_rewrite(self, query_text: str) -> str:
        model_input = self._build_model_input(query_text)
        logger.debug("LLM rewrite input: %s", model_input)
        inputs = self._tokenizer(model_input, return_tensors="pt", truncation=True)
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated = self._model.generate(
            **inputs,
            max_new_tokens=self._config.max_new_tokens,
            do_sample=False,
        )
        decoded = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        if decoded.startswith(model_input):
            return decoded[len(model_input) :].strip()
        return decoded.strip()

    def _build_model_input(self, query_text: str) -> str:
        prompt = self._config.prompt_template.strip() or DEFAULT_PROMPT_TEMPLATE
        compact_query = query_text.strip().replace("\n", " ")
        return f"Инструкция: {prompt} Запрос: {compact_query} Переписанный запрос:"

    def _parse_candidate(self, raw: str) -> str | None:
        if not raw.strip():
            return None
        first_line = raw.strip().splitlines()[0].strip(" -\t\r\n")
        first_line = first_line[: self._config.max_query_length]
        return first_line or None


__all__ = ["LLMQueryRewriter", "LLMRewriterConfig", "DEFAULT_PROMPT_TEMPLATE"]
