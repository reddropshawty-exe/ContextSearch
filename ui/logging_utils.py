"""Утилиты для настройки логирования приложения."""
from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging() -> None:
    """Настроить логирование в файл и консоль."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level_name = os.getenv("CONTEXTSEARCH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("CONTEXTSEARCH_LOG_FILE", "contextsearch.log")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

