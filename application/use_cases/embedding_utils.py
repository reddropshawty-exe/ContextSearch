"""Вспомогательные функции для работы с набором эмбеддеров."""
from __future__ import annotations

from typing import Mapping

from domain.entities import EmbeddingSpec
from domain.interfaces import Embedder


def get_embedder_for_spec(spec: EmbeddingSpec, embedders: Mapping[str, Embedder]) -> Embedder:
    """Вернуть эмбеддер, соответствующий спецификации."""
    if spec.model_name in embedders:
        return embedders[spec.model_name]
    available = ", ".join(sorted(embedders.keys())) or "нет доступных"
    raise ValueError(
        "Не найден эмбеддер для спецификации "
        f"{spec.id} (model_name={spec.model_name}). Доступные: {available}."
    )


__all__ = ["get_embedder_for_spec"]
