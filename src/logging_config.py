"""Structured logging setup for InSift."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from src.config import Settings, get_settings


RESERVED_LOG_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON without sensitive payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in RESERVED_LOG_KEYS:
                continue
            if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                payload[key] = "***"
            else:
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(settings: Optional[Settings] = None) -> None:
    """Configure root logging for scripts and the Streamlit app."""

    settings = settings or get_settings()
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    fields: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a structured log event with a stable event name."""

    logger.log(level, event, extra=dict(fields or {}))
