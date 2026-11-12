"""Логирование: консоль + файл, структурированный JSON в файле."""
from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler

from backend.core.config import get_settings

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if extra:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    settings.ensure_dirs()
    root = logging.getLogger("mlf")
    root.setLevel(level or settings.log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(console)
    fh = RotatingFileHandler(settings.log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mlf.{name}")
