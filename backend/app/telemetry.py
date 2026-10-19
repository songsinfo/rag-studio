"""Opt-in Langfuse diagnostics sharing the application's OTel provider.

Content requires explicit global and category switches. MySQL remains authoritative.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_configured = False
_provider = None
_client = None
_status = "DISABLED"
_pid = os.getpid()
_batch_var: ContextVar[Optional["ObservationBatch"]] = ContextVar("rag_observation_batch", default=None)
_UNSET = object()
_IO_KEYS = frozenset({"langfuse.observation.input", "langfuse.observation.output"})
_IO_MAX_BYTES = 1024 * 1024


def content_enabled(category: str) -> bool:
    return (category in {"MODEL", "RETRIEVAL", "TOOL", "QA"}
            and os.getenv("RAG_LANGFUSE_DATA_MODE", "METADATA_ONLY").strip().upper() == "CONTENT"
            and os.getenv("RAG_LANGFUSE_CAPTURE_" + category, "true").strip().lower() in {"1", "true", "yes"})


def _content_category(name):
    if name.startswith("model.") or name == "agent.model":
        return "MODEL"
    if name.startswith("rag."):
        return "RETRIEVAL"
    if name == "agent.tool":
        return "TOOL"
    if name in {"qa.worker.attempt", "agent.run"}:
        return "QA"
    return ""


def _content_json(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    size = len(encoded.encode("utf-8"))
    if size > _IO_MAX_BYTES:
        return json.dumps({"captureStatus": "OMITTED_TOO_LARGE", "originalBytes": size,
                           "limitBytes": _IO_MAX_BYTES})
    return encoded


def set_span_io(category, *, span=None, input=_UNSET, output=_UNSET):
    """Only explicit content fields; never inspect runtime objects or log bodies."""
    if not content_enabled(category) or not export_current_span():
        return
    try:
        from opentelemetry import trace
        span = span if span is not None else trace.get_current_span()
        if span is None or not span.is_recording() or _content_category(span.name) != category:
            return
        for field, value in (("input", input), ("output", output)):
            if value is not _UNSET:
                span.set_attribute("langfuse.observation." + field, _content_json(value))
    except Exception:
        logger.warning("telemetry_content_capture_failed")


def model_response_content(result):
    """Response body only, excluding HTTP headers and provider metadata."""
    try:
        if isinstance(result, (dict, list, str)):
            return result
        if hasattr(result, "content"):
            return {"content": result.content, "tool_calls": getattr(result, "tool_calls", []),
                    "invalid_tool_calls": getattr(result, "invalid_tool_calls", [])}
        if callable(getattr(result, "json", None)):
            return result.json()
    except Exception:
        pass
    return {"captureStatus": "UNSUPPORTED_RESPONSE_TYPE"}


def set_model_io(observation, *, input=_UNSET, output=_UNSET):
    if observation is None or not content_enabled("MODEL") or not export_current_span():
        return
    try:
        observation.update(**{key: json.loads(_content_json(value))
            for key, value in (("input", input), ("output", output)) if value is not _UNSET})
    except Exception:
        logger.warning("telemetry_model_content_capture_failed")


class _MetadataOnlyExporter:
    """Keep optional Langfuse bodies out of the pre-existing OTLP destination."""
    def __init__(self, exporter):
        self.exporter = exporter

    def export(self, spans):
        from opentelemetry.sdk.trace import ReadableSpan
        cleaned = [ReadableSpan(name=s.name, context=s.context, parent=s.parent,
            resource=s.resource, attributes={k: v for k, v in s.attributes.items() if k not in _IO_KEYS},
            events=s.events, links=s.links, kind=s.kind, status=s.status,
            start_time=s.start_time, end_time=s.end_time, instrumentation_scope=s.instrumentation_scope)
            for s in spans]
        return self.exporter.export(cleaned)

    def shutdown(self):
        return self.exporter.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self.exporter.force_flush(timeout_millis)


@contextmanager
def safe_span(name, attributes=None, *, context=None):
    """Catch tracing setup/cleanup faults without catching business exceptions."""
    scope = None
    span = None
    try:
        from opentelemetry import trace
        scope = trace.get_tracer("rag.agent").start_as_current_span(
            name, context=context, attributes=safe_attributes(attributes),
            record_exception=False, set_status_on_exception=False,
        )
        span = scope.__enter__()
    except Exception:
        scope = None
        logger.warning("telemetry_span_setup_failed")
    try:
        yield span
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except Exception:
                logger.warning("telemetry_span_cleanup_failed")

_HASH_RE = re.compile(r"^[0-9a-f]{32}$")
_CODE_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:/-]{0,63}$")
_SAFE_KEYS = frozenset({
    "rag.export", "rag.environment", "rag.hmac_version", "rag.tenant_hmac",
    "rag.run_hmac", "rag.job_hmac", "rag.conversation_hmac", "rag.segment_hmac",
    "rag.lease_epoch", "rag.job_attempt", "rag.retrieval_index", "rag.hit_count",
    "rag.duration_ms", "rag.input_count", "rag.used_bytes", "rag.max_bytes",
    "rag.status", "rag.error_code", "rag.model_alias", "rag.purpose",
    "rag.call_hmac", "rag.stage", "rag.enabled", "rag.route_count",
    "agent.round", "agent.prompt_version", "tool.name", "tool.category",
    "tool.mock",
})
_HASH_KEYS = frozenset({"rag.tenant_hmac", "rag.run_hmac", "rag.job_hmac",
                        "rag.conversation_hmac", "rag.segment_hmac", "rag.call_hmac"})
_NUMBER_KEYS = frozenset({"rag.lease_epoch", "rag.job_attempt", "rag.retrieval_index",
                          "rag.hit_count", "rag.duration_ms", "rag.input_count",
                          "rag.used_bytes", "rag.max_bytes", "rag.route_count",
                          "agent.round"})
_ALLOWED_EVENTS = frozenset({"guardrail.indirect_prompt_injection", "model.not_sent"})


@dataclass(frozen=True)
class ObservationBatch:
    trace_id: Optional[str]
    sampled: bool
    target: str
    job_attempt: int
    lease_epoch: int
    hmac_version: str
    tenant_id: Optional[str] = None


def _enabled() -> bool:
    return os.getenv("RAG_LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


_SAFE_TOOLS = frozenset({
    "search_knowledge", "read_section", "diagnose_coupon",
    "get_final_price_rules", "diagnose_product_group", "diagnose_subsidy",
})


def _safe_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value if _CODE_RE.fullmatch(value) else None


def safe_attributes(attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed: permit named, typed scalar metadata, never arbitrary strings."""
    result: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if key not in _SAFE_KEYS or value is None:
            continue
        if key in _HASH_KEYS:
            if isinstance(value, str) and _HASH_RE.fullmatch(value):
                result[key] = value
        elif key in _NUMBER_KEYS:
            if type(value) in {int, float} and 0 <= value < 1e12:
                result[key] = value
        elif key in {"rag.export", "rag.enabled", "tool.mock"}:
            if type(value) is bool:
                result[key] = value
        elif key == "tool.name":
            if value in _SAFE_TOOLS:
                result[key] = value
        elif isinstance(value, str) and _safe_code(value):
            result[key] = value
    return result


def identifier(kind: str, value: Any, tenant_id: Optional[str] = None) -> Optional[str]:
    """Independent, versioned HMAC; never fall back to unkeyed or plaintext IDs."""
    key = os.getenv("RAG_LANGFUSE_ID_HMAC_KEY", "")
    version = os.getenv("RAG_LANGFUSE_HMAC_VERSION", "v1")
    env = os.getenv("RAG_ENVIRONMENT", "demo")
    if len(key.encode("utf-8")) < 32 or not all(_safe_code(x) for x in (kind, version, env)):
        return None
    batch = active_batch()
    namespace = tenant_id or (batch.tenant_id if batch else "") or ""
    return hmac.new(key.encode("utf-8"),
                    (version + "|" + env + "|" + namespace + "|" + kind + "|" + str(value)).encode("utf-8"),
                    hashlib.sha256).hexdigest()[:32]


# SDK 4.16.0 flattens dict metadata into individual OTel attributes.
# Never accept a free-form metadata JSON blob or arbitrary SDK attributes.
_SDK_METADATA_CODES = frozenset({
    "purpose", "status", "errorCode", "costCurrency", "costStatus", "usageSource", "cacheStatus",
})
_SDK_METADATA_HASHES = frozenset({"call"})
_SDK_METADATA_NUMBERS = frozenset({"durationMs", "settledCny"})
_SDK_METADATA_PREFIX = "langfuse.observation.metadata."
_SDK_ALLOWED = frozenset({
    "langfuse.observation.type", "langfuse.observation.model.name",
    "langfuse.observation.usage_details",
    "langfuse.internal.is_app_root", "langfuse.internal.as_root",
    "langfuse.environment", "langfuse.release", "session.id",
}) | frozenset(
    _SDK_METADATA_PREFIX + k for k in
    (_SDK_METADATA_CODES | _SDK_METADATA_HASHES | _SDK_METADATA_NUMBERS)
)


def _safe_sdk_value(key: str, value: Any, span_name: str = "") -> bool:
    if key in _IO_KEYS:
        if not content_enabled(_content_category(span_name)) or not isinstance(value, str):
            return False
        try:
            json.loads(value)
            return len(value.encode("utf-8")) <= _IO_MAX_BYTES
        except (ValueError, TypeError):
            return False
    if key.startswith(_SDK_METADATA_PREFIX):
        name = key[len(_SDK_METADATA_PREFIX):]
        if name in _SDK_METADATA_CODES:
            return bool(_safe_code(value))
        if name in _SDK_METADATA_HASHES:
            return isinstance(value, str) and bool(_HASH_RE.fullmatch(value))
        if name in _SDK_METADATA_NUMBERS:
            # Langfuse serializes floats to JSON numeric strings on update().
            if type(value) in {int, float}:
                return 0 <= value < 1e12 and value == value
            return (isinstance(value, str)
                    and re.fullmatch(r"(?:0|[1-9][0-9]{0,11})(?:[.][0-9]+)?(?:e-[0-9]+)?", value) is not None
                    and 0 <= float(value) < 1e12)
        return False
    if key == "langfuse.observation.type":
        return value in {"generation", "embedding"}
    if key == "langfuse.observation.model.name":
        return bool(_safe_code(value))
    if key == "langfuse.observation.usage_details":
        try:
            usage = json.loads(value) if isinstance(value, str) else value
            return isinstance(usage, dict) and all(
                k in {"input", "output", "cache_read_input_tokens"} and type(v) is int
                and 0 <= v <= 100000000 for k, v in usage.items())
        except (TypeError, ValueError):
            return False
    if key == "session.id":
        return isinstance(value, str) and bool(_HASH_RE.fullmatch(value))
    if key in {"langfuse.environment", "langfuse.release"}:
        return bool(_safe_code(value))
    if key in {"langfuse.internal.is_app_root", "langfuse.internal.as_root"}:
        return type(value) is bool
    return False


def _export_span(span: Any) -> bool:
    """Strict scope + attributes/events allowlist, including SDK generation spans."""
    try:
        scope = span.instrumentation_scope.name
        if scope == "langfuse-sdk":
            if not str(span.name).startswith("model.") or span.events or getattr(span.status, "description", None):
                return False
            return all(key in (_SDK_ALLOWED | _IO_KEYS) and _safe_sdk_value(key, value, span.name)
                       for key, value in dict(span.attributes or {}).items())
        if scope != "rag.agent" or not str(span.name).startswith(("qa.", "agent.", "rag.")):
            return False
        attrs = dict(span.attributes or {})
        if attrs.get("rag.export") is not True:
            return False
        project_attrs = {k: v for k, v in attrs.items() if k not in (_SDK_ALLOWED | _IO_KEYS)}
        sdk_attrs = {k: v for k, v in attrs.items() if k in (_SDK_ALLOWED | _IO_KEYS)}
        if (len(safe_attributes(project_attrs)) != len(project_attrs)
                or not all(_safe_sdk_value(k, v, span.name) for k, v in sdk_attrs.items())
                or getattr(span.status, "description", None)):
            return False
        for event in span.events or ():
            if event.name not in _ALLOWED_EVENTS:
                return False
            if len(safe_attributes(dict(event.attributes or {}))) != len(event.attributes or {}):
                return False
        return True
    except Exception:
        return False


def _mask_otel_spans(*, params):
    """Defense in depth; the export filter already rejects unknown events/scopes.

    Raw OTel events cannot be patched by the SDK, hence _export_span rejects them.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    # Named content fields have a separate opt-in policy; arbitrary metadata,
    # third-party payload attributes and exception data remain excluded.
    patches = {}
    for ref, span in params.spans.items():
        attrs = span.attributes
        is_sdk = span.instrumentation_scope_name == "langfuse-sdk"
        if is_sdk:
            to_delete = tuple(key for key, value in attrs.items() if key not in (_SDK_ALLOWED | _IO_KEYS) or not _safe_sdk_value(key, value, span.name))
        else:
            allowed = {**safe_attributes(attrs), **{
                k: v for k, v in attrs.items() if k in (_SDK_ALLOWED | _IO_KEYS) and _safe_sdk_value(k, v, span.name)
            }}
            to_delete = tuple(key for key in attrs if key not in allowed)
        if to_delete:
            patches[ref] = OtelSpanPatch(delete_attributes=to_delete)
    return MaskOtelSpansResult(span_patches=patches)


def configure_telemetry() -> None:
    """One provider per process; no duplicate processor when embedded workers start."""
    global _configured, _provider, _client, _status, _pid
    with _lock:
        if _pid != os.getpid():
            _configured, _provider, _client = False, None, None
            _pid = os.getpid()
        if _configured:
            return
        _configured = True
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        lf_requested = _enabled()
        if not endpoint and not lf_requested:
            _status = "DISABLED"
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                candidate = TracerProvider(shutdown_on_exit=False, resource=Resource.create({
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "rag-backend")
                }))
                trace.set_tracer_provider(candidate)
                provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                _status = "PROVIDER_UNAVAILABLE"
                return
            _provider = provider
            if endpoint:
                exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=2)
                provider.add_span_processor(BatchSpanProcessor(
                    _MetadataOnlyExporter(exporter), max_queue_size=512, max_export_batch_size=128,
                    schedule_delay_millis=1000, export_timeout_millis=2000))
            if not lf_requested:
                _status = "OTLP_ONLY"
                return
            if not all(os.getenv(k, "").strip() for k in (
                    "LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
                _status = "MISSING_CONFIG"
                logger.warning("Langfuse disabled: missing configuration")
                return
            from langfuse import Langfuse
            # v4 attaches a single LangfuseSpanProcessor to the supplied shared provider.
            _client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                base_url=os.environ["LANGFUSE_BASE_URL"],
                tracer_provider=provider,
                sample_rate=1.0,  # batch-level sampling is decided before spans start
                timeout=2,
                flush_at=128,
                flush_interval=1,
                environment=os.getenv("RAG_ENVIRONMENT", "demo"),
                release="git-" + os.getenv("RAG_COMMIT_SHA", "unknown"),
                media_upload_thread_count=1,
                should_export_span=_export_span,
                mask_otel_spans=_mask_otel_spans,
            )
            _status = "ENABLED"
            logger.info("Langfuse tracing enabled")
        except Exception:
            _client = None
            _status = "INIT_FAILED"
            logger.warning("Langfuse/OTel initialization failed; business execution unaffected")


def telemetry_status() -> str:
    return _status


def active_batch() -> Optional[ObservationBatch]:
    return _batch_var.get()


def configured_client():
    return _client if _status == "ENABLED" and _enabled() else None


def export_current_span() -> bool:
    batch = active_batch()
    return bool(batch and batch.sampled and configured_client() is not None)


@contextmanager
def observation_batch(job: dict[str, Any]) -> Iterator[Optional[ObservationBatch]]:
    """New physical OTel trace per claimed QA lease; never a span across HITL/restart."""
    if job.get("kind") not in {"QA_RUN", "EVALUATE"} or configured_client() is None:
        yield None
        return
    try:
        rate = float(os.getenv("RAG_LANGFUSE_SAMPLE_RATE", "1"))
        if not (0 <= rate <= 1):
            raise ValueError("bad sample rate")
    except ValueError:
        rate = 0.0  # configuration error fails closed
    sampled = secrets.randbelow(1000000) < int(rate * 1000000)
    attempt = int(job.get("attempt") or 0) + 1
    epoch = int(job.get("lease_epoch") or 0)
    target = _safe_code(os.getenv("RAG_LANGFUSE_TARGET_ID", "private")) or "private"
    version = _safe_code(os.getenv("RAG_LANGFUSE_HMAC_VERSION", "v1")) or "v1"
    from opentelemetry.context import Context
    # Disabled sample still has a LOCAL mapping event; no raw IDs are exported.
    root_name = "qa.worker.attempt" if job["kind"] == "QA_RUN" else "agent.evaluation.attempt"
    with safe_span(root_name, {
            "rag.export": sampled,
            "rag.job_attempt": attempt, "rag.lease_epoch": epoch,
            "rag.environment": os.getenv("RAG_ENVIRONMENT", "demo"),
            "rag.hmac_version": version,
            "rag.tenant_hmac": identifier("tenant", job.get("tenant_id"), job.get("tenant_id")),
            "rag.run_hmac": identifier("run", job.get("target_id"), job.get("tenant_id")),
            "rag.job_hmac": identifier("job", job.get("id"), job.get("tenant_id")),
        }, context=Context()) as span:
        trace_id = format(span.get_span_context().trace_id, "032x") if span and span.is_recording() else None
        batch = ObservationBatch(trace_id, bool(sampled and trace_id), target, attempt, epoch, version,
                                 job.get("tenant_id"))
        token = _batch_var.set(batch)
        try:
            yield batch
        finally:
            _batch_var.reset(token)


@contextmanager
def qa_identity_scope(conversation_id: str, segment_id: str) -> Iterator[None]:
    """Update root + propagate HMAC session/identities to Langfuse children."""
    batch = active_batch()
    if batch is None or not batch.sampled:
        yield
        return
    from opentelemetry import trace
    attrs = safe_attributes({
        "rag.conversation_hmac": identifier("conversation", conversation_id),
        "rag.segment_hmac": identifier("segment", segment_id),
    })
    try:
        for key, val in attrs.items():
            trace.get_current_span().set_attribute(key, val)
    except Exception:
        logger.warning("Langfuse identity propagation unavailable")
    session_id = identifier("conversation", conversation_id)
    if session_id is None:
        yield
        return
    scope = None
    try:
        from langfuse import propagate_attributes
        scope = propagate_attributes(session_id=session_id)
        scope.__enter__()
    except Exception:
        scope = None
        logger.warning("Langfuse session propagation unavailable")
    try:
        yield
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except Exception:
                logger.warning("Langfuse session cleanup unavailable")


@contextmanager
def model_observation(purpose: str, alias: str, call_id: str) -> Iterator[Any]:
    """One manually ended generation for each genuinely sent provider attempt."""
    if not export_current_span():
        yield None
        return
    try:
        name = "model." + (_safe_code(str(purpose).lower()) or "unknown")
        model = _safe_code(alias) or "configured"
        metadata = {"purpose": _safe_code(purpose) or "UNKNOWN"}
        call_hash = identifier("call", call_id)
        if call_hash:
            metadata["call"] = call_hash
        observation = _client.start_observation(as_type="generation", name=name,
                                                 model=model, metadata=metadata)
    except Exception:
        logger.warning("Langfuse generation start failed")
        yield None
        return
    try:
        yield observation
    finally:
        pass  # finished after provider usage is known by finish_model_observation


def finish_model_observation(observation: Any, *, status: str, duration_ms: float,
                             reported_usage: Optional[dict] = None,
                             settlement: Optional[dict] = None,
                             usage_source: str = "UNKNOWN",
                             error_code: Optional[str] = None) -> None:
    if observation is None:
        return
    try:
        metadata: dict[str, Any] = {"status": _safe_code(status) or "UNKNOWN",
                                     "durationMs": round(max(0.0, duration_ms), 2),
                                     "usageSource": _safe_code(usage_source) or "UNKNOWN",
                                     "cacheStatus": "REPORTED" if reported_usage and
                                         "cache_read_input_tokens" in reported_usage else "UNKNOWN",
                                     "costStatus": "UNKNOWN"}
        if error_code:
            metadata["errorCode"] = _safe_code(error_code) or "UNKNOWN"
        if settlement:
            metadata["costCurrency"] = "CNY"
            metadata["costStatus"] = "SETTLED" if settlement.get("settledAmount") is not None else "UNKNOWN"
            if settlement.get("settledAmount") is not None:
                # CNY is not automatically converted to Langfuse's native cost currency.
                try:
                    metadata["settledCny"] = round(float(settlement["settledAmount"]), 6)
                except (ValueError, TypeError, OverflowError):
                    pass
        kwargs: dict[str, Any] = {"metadata": metadata}
        if reported_usage:
            tokens = {k: v for k, v in reported_usage.items() if
                      k in {"input", "output", "cache_read_input_tokens"} and type(v) is int and v >= 0}
            if tokens:
                kwargs["usage_details"] = tokens
        observation.update(**kwargs)
    except Exception:
        logger.warning("Langfuse generation update failed")
    finally:
        try:
            observation.end()
        except Exception:
            logger.warning("Langfuse generation end failed")


def shutdown_telemetry(timeout_seconds: float = 2.0) -> None:
    """Bound process exit; exporter failure must never block job completion."""
    global _client, _status
    with _lock:
        client, provider = _client, _provider
        _client = None
        # The global OTel provider cannot be reset safely after shutdown.
        # Repeated API lifespan calls therefore never append new processors.
    def close():
        try:
            if client:
                # The SDK unregisters its blocking atexit hook before flushing.
                client.shutdown()
            if provider:
                provider.force_flush(timeout_millis=1000)
                provider.shutdown()
        except Exception:
            logger.warning("Telemetry flush/shutdown failed")
    if client or provider:
        thread = threading.Thread(target=close, name="rag-telemetry-close", daemon=True)
        thread.start()
        thread.join(timeout=max(0.0, min(float(timeout_seconds), 10.0)))
        if thread.is_alive():
            _status = "SHUTDOWN_TIMEOUT"
            logger.warning("Telemetry shutdown timed out; spans may be dropped")
        else:
            _status = "STOPPED"
