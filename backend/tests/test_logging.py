import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter, configure_logging, log_context, log_stage


@pytest.fixture
def logs(monkeypatch):
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpx2", "httpcore", "openai")
    states = {name: (list(logging.getLogger(name).handlers), logging.getLogger(name).level,
                     logging.getLogger(name).propagate) for name in names}
    monkeypatch.delenv("RAG_LOG_FILE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield stream
    for added in list(root.handlers):
        if added not in handlers:
            root.removeHandler(added)
            added.close()
    root.setLevel(level)
    for name, (saved_handlers, saved_level, propagate) in states.items():
        logger = logging.getLogger(name)
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = propagate


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_logs_redact_keys_and_exception_content(logs, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", 'secret-"key')
    logger = logging.getLogger("app.test")
    try:
        raise ValueError("private document text and secret-key")
    except ValueError:
        with log_context(run_id="run-1"):
            logger.exception('failed secret-"key', extra={"fields": {"error_code": "PARSE_FAILED"}})
    data = events(logs)[-1]
    assert data["run_id"] == "run-1"
    assert data["error_code"] == "PARSE_FAILED"
    assert data["exceptions"][0]["type"] == "ValueError"
    assert data["exceptions"][0]["frames"][-1]["function"] == "test_json_logs_redact_keys_and_exception_content"
    assert 'secret-' not in logs.getvalue()
    assert "private document text" not in logs.getvalue()


def test_concurrent_contexts_and_stage_timing_do_not_leak(logs, monkeypatch):
    from app import logging_config
    times = iter([1.0, 1.125])
    monkeypatch.setattr(logging_config.time, "perf_counter", lambda: next(times))
    logger = logging.getLogger("app.test")
    with log_context(run_id="timed"):
        log_stage(logger, "run_stage", "RETRIEVING", run_id="timed")
        log_stage(logger, "run_stage", "GENERATING", run_id="timed")
    assert events(logs)[-1]["duration_ms"] == 125
    assert events(logs)[-1]["previous_stage"] == "RETRIEVING"

    def emit(number):
        with log_context(job_id=f"job-{number}"):
            logger.info("worker")
        logger.info("outside")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(emit, [1, 2]))
    assert {row["job_id"] for row in events(logs) if row["event"] == "worker"} == {"job-1", "job-2"}
    assert all("job_id" not in row and "run_id" not in row for row in events(logs) if row["event"] == "outside")


def test_file_logging_is_idempotent_and_rotates(logs, monkeypatch, tmp_path):
    path = tmp_path / "runtime" / "backend.log"
    monkeypatch.setenv("RAG_LOG_FILE", str(path))
    configure_logging()
    configure_logging()
    owned = [handler for handler in logging.getLogger().handlers if getattr(handler, "rag_handler", False)]
    assert len(owned) == 2
    file_handler = next(handler for handler in owned if hasattr(handler, "maxBytes"))
    file_handler.maxBytes = 1
    logging.getLogger("app.test").info("first")
    logging.getLogger("app.test").info("second")
    assert json.loads(path.read_text(encoding="utf-8"))["event"] == "second"
    assert json.loads(path.with_name("backend.log.1").read_text(encoding="utf-8"))["event"] == "first"


def test_request_ids_cover_csrf_rejections_without_logging_request_body(client, logs):
    configure_logging()
    first = client.get("/api/v1/health?api_key=private-query")
    second = client.post("/api/v1/auth/login", json={"password": "private-password"})
    assert first.status_code == 200 and second.status_code == 403
    assert first.headers["x-request-id"] != second.headers["x-request-id"]
    recorded = [row for row in events(logs) if row["event"] == "request_finished"]
    assert [row["status"] for row in recorded] == [200, 403]
    assert recorded[0]["request_id"] == first.headers["x-request-id"]
    assert "private-query" not in logs.getvalue() and "private-password" not in logs.getvalue()


def test_unhandled_error_has_request_id_and_safe_stack(logs):
    from app.main import request_logging_middleware, generic_exception_handler
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)
    app.add_exception_handler(Exception, generic_exception_handler)

    @app.get("/fail")
    def fail():
        raise RuntimeError("private model response")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/fail")
    assert response.status_code == 500
    row = next(row for row in events(logs) if row["event"] == "request_failed")
    assert row["request_id"] == response.headers["x-request-id"]
    assert row["exceptions"] and "private model response" not in logs.getvalue()


@pytest.mark.parametrize("failed", [False, True])
def test_model_calls_log_usage_and_failure_without_payload(logs, monkeypatch, failed):
    from app.qa.model_gateway import ModelGateway
    from app.qa.model_usage import invoke_tracked
    from app.model_budget import BudgetService
    from app.errors import ApiException
    import app.qa.model_usage as usage
    monkeypatch.setattr(BudgetService, "reserve", lambda *a, **kw: None)
    monkeypatch.setattr(usage, "get_run_usage", lambda *a: {"modelCalls": 0, "totalTokens": 0})
    monkeypatch.setattr(ModelGateway, "assert_run_sources_current", lambda *a, **kw: None)
    monkeypatch.setattr(ModelGateway, "assert_job_current", lambda *a: None)
    monkeypatch.setattr(ModelGateway, "_record_call", lambda *a, **kw: None)

    def provider():
        if failed:
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", "private document content")
        return "private answer"
    kwargs = dict(tenant_id="tenant-demo-1", run_id="run-model", job_id="job-model", purpose="ANSWER",
                  alias="chat-main", payload={"content": "private document content"},
                  usage_reader=lambda result: {"input_tokens": 12, "output_tokens": 3})
    if failed:
        with pytest.raises(ApiException):
            invoke_tracked(provider, **kwargs)
    else:
        invoke_tracked(provider, **kwargs)
    row = events(logs)[-1]
    assert row["run_id"] == "run-model" and row["job_id"] == "job-model" and row["duration_ms"] >= 0
    assert row["event"] == ("model_call_failed" if failed else "model_call_finished")
    if not failed:
        assert row["input_tokens"] == 12 and row["output_tokens"] == 3
    assert "private document content" not in logs.getvalue() and "private answer" not in logs.getvalue()


@pytest.mark.parametrize("failed", [False, True])
def test_worker_logs_actual_completion_and_retries(logs, monkeypatch, failed, setup_test_db):
    from app.runtime.worker import DurableJobWorker
    from app.models import job_table
    from app.errors import ApiException
    from sqlalchemy import update
    now = datetime.now(timezone.utc)
    with setup_test_db.begin() as conn:
        conn.execute(job_table.insert().values(tenant_id="tenant-demo-1", id="log-job", kind="QA_RUN",
            target_id="log-run", requested_by="user-admin-1", status="QUEUED", stage="ROUTING",
            idempotency_key="logging-job", request_hash=b"0" * 32, created_at=now, updated_at=now))
    worker = DurableJobWorker(worker_id="log-worker")
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda *a: None)

    def execute(job):
        if failed:
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", "private upstream response")
        with setup_test_db.begin() as conn:
            conn.execute(update(job_table).where(job_table.c.id == job["id"]).values(status="COMPLETED"))
    monkeypatch.setattr(worker, "_run_qa", execute)
    assert worker.run_once()
    rows = events(logs)
    finished = next(row for row in rows if row["event"] == "job_finished")
    assert finished["run_id"] == "log-run" and finished["job_id"] == "log-job"
    assert finished["status"] == ("QUEUED" if failed else "COMPLETED")
    if failed:
        assert any(row["event"] == "job_retry_scheduled" for row in rows)
    assert "private upstream response" not in logs.getvalue()
