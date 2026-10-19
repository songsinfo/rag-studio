"""Metadata-only observability regression checks (no network or model calls)."""
import os
from types import SimpleNamespace

from app import telemetry


def _span(name="qa.worker.attempt", scope="rag.agent", attrs=None, events=()):
    return SimpleNamespace(
        name=name,
        instrumentation_scope=SimpleNamespace(name=scope),
        attributes=attrs or {"rag.export": True, "rag.hit_count": 3},
        events=events,
        status=SimpleNamespace(description=None),
    )


def test_disabled_without_explicit_switch(monkeypatch):
    monkeypatch.delenv("RAG_LANGFUSE_ENABLED", raising=False)
    assert telemetry._enabled() is False


def test_hmac_is_versioned_and_scoped(monkeypatch):
    monkeypatch.setenv("RAG_LANGFUSE_ID_HMAC_KEY", "x" * 40)
    monkeypatch.setenv("RAG_LANGFUSE_HMAC_VERSION", "v1")
    monkeypatch.setenv("RAG_ENVIRONMENT", "test")
    a = telemetry.identifier("run", "secret-run-id")
    b = telemetry.identifier("tenant", "secret-run-id")
    assert a and len(a) == 32 and a != b and "secret" not in a
    monkeypatch.setenv("RAG_LANGFUSE_HMAC_VERSION", "v2")
    assert telemetry.identifier("run", "secret-run-id") != a
    monkeypatch.delenv("RAG_LANGFUSE_ID_HMAC_KEY", raising=False)
    assert telemetry.identifier("run", "secret-run-id") is None


def test_metadata_allowlist_fails_closed():
    assert telemetry.safe_attributes({
        "rag.hit_count": 2,
        "rag.error_code": "BUDGET_EXCEEDED",
        "rag.call_hmac": "a" * 32,
        "rag.job_hmac": "source-document-id",
        "tool.call_id": "raw-tool-uuid",
        "exception.message": "secret stack",
        "rag.model_alias": "evil key with spaces!",
    }) == {"rag.hit_count": 2, "rag.error_code": "BUDGET_EXCEEDED",
           "rag.call_hmac": "a" * 32}


def test_scope_and_exception_events_fail_closed():
    assert telemetry._export_span(_span())
    assert not telemetry._export_span(_span(scope="sqlalchemy"))
    assert not telemetry._export_span(_span(attrs={
        "rag.export": True, "gen_ai.prompt": "confidential contents",
    }))
    assert not telemetry._export_span(_span(events=(
        SimpleNamespace(name="exception", attributes={"exception.message": "secret"}),
    )))
    assert not telemetry._export_span(_span(events=(
        SimpleNamespace(name="guardrail.indirect_prompt_injection",
                        attributes={"evidence.id": "E999"}),
    )))


def test_independent_batches_are_context_local():
    assert telemetry.active_batch() is None
    with telemetry.observation_batch({"kind": "INGEST"}):
        assert telemetry.active_batch() is None
    assert telemetry.active_batch() is None


def test_sdk_json_metadata_and_usage_must_be_allowlisted():
    import json
    valid = {
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": "chat-main",
        "langfuse.observation.metadata.purpose": "AGENT",
        "langfuse.observation.metadata.call": "b"*32,
        "langfuse.observation.metadata.durationMs": "10.5",
        "langfuse.observation.usage_details": json.dumps({
            "input": 31, "output": 0}),
    }
    assert telemetry._export_span(_span(name="model.agent", scope="langfuse-sdk", attrs=valid))
    for key, value in [
        ("langfuse.observation.metadata.prompt", "TOP_SECRET"),
        ("langfuse.observation.usage_details", '{"input":10,"raw_input":"SECRET"}'),
        ("langfuse.observation.input", "SENSITIVE_PROMPT"),
    ]:
        unsafe = dict(valid, **{key: value})
        assert not telemetry._export_span(_span(name="model.agent", scope="langfuse-sdk", attrs=unsafe))


def test_candidate_snapshot_is_ordered_and_marks_truncation():
    from app.evaluation.diagnostics import candidate_snapshot
    candidates = [
        {"chunkId": "a", "buildId": "b1", "documentId": "d1",
         "text": "SYNTHETIC_PRIVATE_CONTENT_NEVER_EXPORT",
         "sourceUnits": [{"unitId": "u1", "contextStart": 0, "contextEnd": 10,
                          "contextMapped": True, "location": {"page": 2}}]},
        {"chunkId": "c", "buildId": "b2", "documentId": "d2", "sourceUnits": []},
    ]
    snap = candidate_snapshot("rerank", candidates, limit=1, retrieval_index=7)
    assert snap["complete"] is False
    assert snap["capturedCount"] == 1
    assert snap["items"][0]["rank"] == 1
    assert snap["items"][0]["chunkId"] == "a"
    assert "SYNTHETIC_PRIVATE_CONTENT_NEVER_EXPORT" not in str(snap)
    assert all("text" not in item for item in snap["items"])
    assert snap["items"][0]["sourceUnits"][0]["contextStart"] == 0
    assert candidate_snapshot("context_budget", candidates[:1], limit=1,
                              retrieval_index=7, context=True)["complete"]


def test_score_projection_retains_unknown_and_human_revisions():
    from app.evaluation.langfuse_sync import _create_scores, score_id
    item = {
        "automaticJudgment": {"correctness": "UNDETERMINED",
                              "answerRelevance": "RELEVANT"},
        "metrics": {"hit": 0.0, "recall": None, "ndcg": 0.7,
                    "costCny": None},
    }
    reviews = [{"id": "r1", "labels": {"correctness": "CORRECT"}},
               {"id": "r2", "labels": {"correctness": "WRONG"}}]
    scores = list(_create_scores(item, reviews, "a"*32, "project", "v2"))
    assert any(s["value"] == "UNDETERMINED" for s in scores)
    assert len([s for s in scores if s["name"] == "human.correctness.v2"]) == 2
    assert not any(s["name"].endswith("recall.v1") for s in scores)
    assert any(s["value"] == 0.0 and s["name"] == "automatic.hit.v1" for s in scores)
    assert score_id("p", "r", "correctness", "v2", "r1") == score_id(
        "p", "r", "correctness", "v2", "r1")
    assert score_id("p", "r", "correctness", "v2", "r1") != score_id(
        "p", "r", "correctness", "v2", "r2")


def test_final_trace_mapping_never_falls_back_to_old_sampled_attempt():
    from app.evaluation.langfuse_sync import _final_batch_trace_id
    final_unsampled = SimpleNamespace(payload={
        "traceId": None, "sampled": False, "targetInstance": "private",
        "leaseEpoch": 2,
    })
    older_sampled = SimpleNamespace(payload={
        "traceId": "a"*32, "sampled": True, "targetInstance": "private",
        "leaseEpoch": 1,
    })
    assert _final_batch_trace_id([final_unsampled, older_sampled], "private") is None
    assert _final_batch_trace_id([older_sampled], "private") == "a"*32
    assert _final_batch_trace_id([older_sampled], "another-project") is None


def test_dense_and_bm25_routes_are_captured_before_fusion(monkeypatch):
    from app.retrieval.rag_graph import RagRetrievalGraph
    import app.evaluation.diagnostics as diagnostics
    captured = []

    def store(ctx, stage, hits, limit, *, context=False, route_index=None):
        captured.append((ctx.retrieval_index, stage, route_index,
                         [hit["chunkId"] for hit in hits[:limit]]))

    monkeypatch.setattr(diagnostics, "capture_eval_stage", store)
    runtime = SimpleNamespace(context=SimpleNamespace(retrieval_index=9, observation_context=None))
    cfg = SimpleNamespace(dense_top_k=5, sparse_top_k=5, fusion_top_k=2)
    state = {"cfg": cfg}

    RagRetrievalGraph._observe_stage("dense_recall", lambda s, rt: {
        "dense_routes": [
            [{"chunkId": "gold"}, {"chunkId": "dense-2"}],
            [{"chunkId": "dense-3"}],
        ], "dense_weight": 0.5,
    })(state, runtime)
    RagRetrievalGraph._observe_stage("bm25_recall", lambda s, rt: {
        "sparse_routes": [[{"chunkId": "bm25-gold"}]],
        "sparse_weight": 0.5,
    })(state, runtime)
    RagRetrievalGraph._observe_stage("rrf_fusion", lambda s, rt: {
        "hits": [{"chunkId": "rrf-only"}],
    })(state, runtime)

    assert (9, "dense_recall", 0, ["gold", "dense-2"]) in captured
    assert (9, "dense_recall", 1, ["dense-3"]) in captured
    assert (9, "bm25_recall", 0, ["bm25-gold"]) in captured
    assert (9, "rrf_fusion", None, ["rrf-only"]) in captured
    assert all("gold" not in chunks for _, stage, _, chunks in captured
               if stage == "rrf_fusion")


def test_actual_sdk_generation_span_is_safe_and_parented(monkeypatch):
    """Verify the Langfuse processor/exporter, not just another OTel observer."""
    monkeypatch.setenv("GITHUB_SHA", "9" * 40)
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    langfuse_exporter = InMemorySpanExporter()
    sdk = Langfuse(
        public_key="pk-lf-test-123", secret_key="sk-lf-test-123",
        base_url="http://127.0.0.1:1", tracer_provider=provider,
        span_exporter=langfuse_exporter,
        tracing_enabled=True, flush_at=1000, flush_interval=600,
        should_export_span=telemetry._export_span,
        mask_otel_spans=telemetry._mask_otel_spans,
    )
    try:
        with provider.get_tracer("rag.agent").start_as_current_span(
            "qa.worker.attempt", record_exception=False
        ) as root:
            root.set_attribute("rag.export", True)
            obs = sdk.start_observation(
                as_type="generation", name="model.answer", model="chat-main",
                metadata={"purpose": "ANSWER", "call": "b" * 32},
            )
            obs.update(
                metadata={"status": "SUCCESS", "durationMs": 10.5},
                usage_details={"input": 12, "output": 2},
            )
            obs.end()
            # Must not export unknown span events or unrelated instrumentation.
            with provider.get_tracer("sqlalchemy").start_as_current_span("db.query") as ignored:
                ignored.set_attribute("db.statement", "SENSITIVE_SQL_CANARY")
        sdk.flush()
        exported = list(langfuse_exporter.get_finished_spans())
        assert {s.name for s in exported} == {"qa.worker.attempt", "model.answer"}
        model_span = next(s for s in exported if s.name == "model.answer")
        assert model_span.context.trace_id == root.get_span_context().trace_id
        assert model_span.parent.span_id == root.get_span_context().span_id
        attrs = dict(model_span.attributes or {})
        assert attrs["langfuse.release"] == "synthetic-tests"
        assert attrs["langfuse.observation.metadata.purpose"] == "ANSWER"
        assert attrs["langfuse.observation.metadata.call"] == "b" * 32
        assert attrs["langfuse.observation.metadata.status"] == "SUCCESS"
        assert "langfuse.observation.usage_details" in attrs
        serialized = str([(s.name, dict(s.attributes or {})) for s in exported])
        assert "SENSITIVE_SQL_CANARY" not in serialized
        assert "langfuse.observation.input" not in attrs
    finally:
        sdk.shutdown()
