"""
logging_config.py  --  central logging setup with rotation.
===========================================================

WHY CENTRAL LOGGING
-------------------
In production you cannot ``print()`` your way to observability. You need
structured, levelled, timestamped logs that go to both the console (for local
dev) and a rotating file (so a long-running service never fills the disk).

This module configures Python's standard ``logging`` once, at startup, from the
values in :class:`~api.config.settings.Settings`:

  * a console handler (human-readable),
  * an optional ``RotatingFileHandler`` (5 MB per file, keep 5 backups by
    default) so logs rotate automatically instead of growing forever,
  * an optional one-line JSON formatter (handy when shipping logs to a system
    like CloudWatch / ELK that parses JSON).

Every module gets its logger via :func:`get_logger`, so the whole app shares
one consistent configuration.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from api.config.settings import Settings

_CONFIGURED = False
_LOGGER_NAMESPACE = "search_api"

_TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    """Format each log record as a single JSON line (one event per line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured extras the caller passed via `extra={...}`.
        for key in ("request_id", "method", "path", "latency_ms", "status_code"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: "Settings") -> logging.Logger:
    """
    Configure the ``search_api`` logger tree once. Safe to call repeatedly
    (subsequent calls are no-ops), so importing modules cannot double-log.
    """
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_NAMESPACE)
    if _CONFIGURED:
        return root

    root.setLevel(settings.log_level)
    root.propagate = False  # don't duplicate into the Python root logger

    formatter: logging.Formatter
    formatter = JsonFormatter() if settings.log_json else logging.Formatter(
        _TEXT_FORMAT, datefmt=_DATE_FORMAT
    )

    # Console handler -- always on.
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler -- optional, disabled automatically if the log dir
    # cannot be created (e.g. a read-only container) so logging never crashes
    # the service.
    if settings.log_to_file:
        try:
            log_path = settings.log_file_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as exc:  # pragma: no cover - defensive
            root.warning("File logging disabled (%s): %s", type(exc).__name__, exc)

    _CONFIGURED = True
    root.debug("Logging configured (level=%s, file=%s, json=%s)",
               settings.log_level, settings.log_to_file, settings.log_json)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the shared ``search_api`` namespace."""
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")


def reset_logging_for_tests() -> None:  # pragma: no cover - test helper
    """Tear down handlers so a fresh configure_logging() can run in tests."""
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_NAMESPACE)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _CONFIGURED = False
