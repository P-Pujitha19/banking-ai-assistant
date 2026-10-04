"""Structured JSON logging with request-id propagation and automatic privacy masking."""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from banking_ai.security.masking import mask_mapping, mask_text

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
LOGGER_NAME = "banking_ai"


class MaskingFilter(logging.Filter):
    """Masks sensitive substrings in every record before it is emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_text(record.getMessage())
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(mask_mapping(fields))
        if record.exc_info:
            payload["exception"] = mask_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None, json_logs: bool = True) -> logging.Logger:
    """Configure the ``banking_ai`` logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter: logging.Formatter = (
        JsonFormatter() if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(MaskingFilter())
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of ``banking_ai``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``event`` with structured key/value ``fields``."""
    logger.log(level, event, extra={"fields": fields})
