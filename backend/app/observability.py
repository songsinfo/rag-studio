"""Whitelisted metadata spans; explicit optional content is added by telemetry."""
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from app.telemetry import export_current_span, safe_attributes


@contextmanager
def agent_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    from app.telemetry import safe_span
    with safe_span(name, {**safe_attributes(attributes), "rag.export": export_current_span()}) as span:
        yield span

def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    if name not in {"guardrail.indirect_prompt_injection", "model.not_sent"}:
        return
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.add_event(name, attributes=safe_attributes(attributes))
    except Exception:
        # Span/event failures are never business failures.
        return
