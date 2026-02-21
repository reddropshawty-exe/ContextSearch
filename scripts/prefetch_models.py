"""Скачать и кешировать модели в локальную директорию для offline-режима."""
from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "google/embeddinggemma-300m",
)
DEFAULT_REWRITER_MODEL = "google/flan-t5-small"


def prefetch_embedding_model(model_name: str, output_dir: Path) -> Path:
    model = SentenceTransformer(model_name)
    target_dir = output_dir / model_name
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target_dir))
    return target_dir


def prefetch_rewriter_model(model_name: str, output_dir: Path) -> Path:
    target_dir = output_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(str(target_dir))
    model.save_pretrained(str(target_dir))
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Директория для сохранения локальных моделей (по умолчанию: models)",
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        dest="embedding_models",
        help="ID embedding-модели из Hugging Face. Можно передавать несколько раз.",
    )
    parser.add_argument(
        "--rewriter-model",
        default=DEFAULT_REWRITER_MODEL,
        help=f"ID модели для query rewriter (по умолчанию: {DEFAULT_REWRITER_MODEL})",
    )
    parser.add_argument(
        "--skip-rewriter",
        action="store_true",
        help="Не скачивать модель query rewriter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    embedding_models = tuple(args.embedding_models or DEFAULT_EMBEDDING_MODELS)

    print(f"Сохраняем модели в: {output_dir}")
    for model_name in embedding_models:
        saved_path = prefetch_embedding_model(model_name, output_dir)
        print(f"embedding: {model_name} -> {saved_path}")

    if not args.skip_rewriter:
        saved_path = prefetch_rewriter_model(args.rewriter_model, output_dir)
        print(f"rewriter: {args.rewriter_model} -> {saved_path}")


if __name__ == "__main__":
    main()
