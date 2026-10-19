"""Structured runtime logs shared by the API and standalone worker."""
import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


_context = ContextVar("log_context", default={})
_phase = ContextVar("log_phase", default=None)


@contextmanager
def log_context(**fields):
    token = _context.set({**_context.get(), **{k: v for k, v in fields.items() if v is not None}})
    phase_token = _phase.set(None)
    try:
        yield
    finally:
        _context.reset(token)
        _phase.reset(phase_token)


def log_stage(logger, event, stage, **fields):
    key = fields.get("run_id") or fields.get("job_id")
    now = time.perf_counter()
    previous = _phase.get()
    details = {**fields, "stage": stage}
    if previous and previous[0] == key:
        details.update(previous_stage=previous[1], duration_ms=round((now - previous[2]) * 1000, 2))
    _phase.set((key, stage, now))
    logger.info(event, extra={"fields": details})


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_context.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            # SQL/provider exceptions can embed prompts, parameters or credentials.
            # Preserve every stack frame and exception type without their raw messages.
            errors = []
            exc = record.exc_info[1]
            seen = set()
            while exc is not None and id(exc) not in seen:
                seen.add(id(exc))
                errors.append({
                    "type": type(exc).__name__,
                    "frames": [
                        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                        for frame in traceback.extract_tb(exc.__traceback__)
                    ],
                })
                exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
            data["exceptions"] = errors
        secrets = [value for name, value in os.environ.items() if value and
                   any(part in name.upper() for part in ("PASSWORD", "SECRET", "API_KEY", "MASTER_KEY"))]

        def redact(value):
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
                return value
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return json.dumps(redact(data), ensure_ascii=False, default=str)


def configure_logging():
    root = logging.getLogger()
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(level)
    # Repeated lifespan starts must not duplicate our handlers or remove test handlers.
    for handler in list(root.handlers):
        if getattr(handler, "rag_handler", False):
            root.removeHandler(handler)
            handler.close()
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = os.getenv("RAG_LOG_FILE", "").strip()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"))
    for handler in handlers:
        handler.rag_handler = True
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Our middleware records access without query strings or duplicated polling logs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
