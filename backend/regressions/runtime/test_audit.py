"""Offline runtime and accounting regression checks."""
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select, update

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
spec = importlib.util.spec_from_file_location("previous_review", ROOT / "backend/regressions/ingestion/test_review.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
setup_test_db = previous.setup_test_db
tmp_path = previous.tmp_path
ACTOR = previous.ACTOR

from app.knowledge.service import KnowledgeService
from app.models import document_table
from app.errors import ApiException
from app.models import qa_run_table, model_call_table, eval_run_table
from app.qa.agent import LangGraphAgent
from app.qa.model_gateway import ModelGateway
from app.qa.service import QaService
from app.retrieval.config import RetrievalConfig
from app.retrieval.milvus_service import MilvusHybridService
from app.runtime.worker import DurableJobWorker
import app.db as db
import app.config as config


def submit_test_run(actor, conv_id, body, **kwargs):
    body = {"segmentId": QaService.get_conversation_detail(actor, conv_id)["currentSegmentId"], **body}
    kwargs.setdefault("idempotency_key", str(uuid.uuid4()))
    return QaService.submit_run(actor, conv_id, body, **kwargs)


def set_prices(alias="chat-main"):
    from app.models import config_revision_table
    with db.engine.begin() as conn:
        conn.execute(update(config_revision_table).where(config_revision_table.c.id == f"cfg-model-{alias}").values(
            payload={"modelAlias": alias, "currency": "CNY", "inputPricePerMillion": 1, "outputPricePerMillion": 1}))


def ingested(monkeypatch, tmp_path):
    doc = previous.queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    return doc


@pytest.mark.parametrize("verdicts", [[], [{"claimId": "other", "status": "SUPPORTED"}],
    [{"claimId": "c1", "status": "SUPPORTED"}] * 2, [{"claimId": "c1", "status": "UNCERTAIN"}]])
def test_strict_rag_rejects_empty_judge_verdicts(monkeypatch, tmp_path, verdicts):
    ingested(monkeypatch, tmp_path)
    real_chat = ModelGateway.chat
    def chat(**kwargs):
        return json.dumps({"verdicts": verdicts}) if kwargs["purpose"] == "JUDGE" else real_chat(**kwargs)
    monkeypatch.setattr(ModelGateway, "chat", chat)
    conv = QaService.create_conversation(ACTOR, {})
    run = submit_test_run(ACTOR, conv["conversationId"], {"question": "Milvus", "strict": True})
    result = QaService.get_run(ACTOR, run["runId"])
    assert result.get("verification") != "SUPPORTED", result


def test_strict_agent_actually_calls_judge(monkeypatch):
    calls = []
    monkeypatch.setattr(LangGraphAgent, "run", lambda **kw: {
        "status": "COMPLETED", "claims": [{"id": "c1", "text": "unverified", "evidence": []}],
        "answer": "unverified", "evidence": [],
    })
    monkeypatch.setattr(ModelGateway, "chat", lambda **kw: calls.append(kw["purpose"]) or '{}')
    conv = QaService.create_conversation(ACTOR, {})
    run = submit_test_run(ACTOR, conv["conversationId"], {"question": "Milvus", "mode": "AGENT", "strict": True})
    result = QaService.get_run(ACTOR, run["runId"])
    assert result.get("verification") != "SUPPORTED" and "JUDGE" in calls, (result, calls)


def test_old_upload_cannot_replace_newer_published_version(monkeypatch, tmp_path):
    old = previous.queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    old_job = worker._claim_one()
    with db.engine.connect() as conn:
        kb_id = conn.execute(select(document_table.c.kb_id)).scalar_one()
    with monkeypatch.context() as patch:
        patch.setattr(config, "is_test_env", lambda: False)
        new = KnowledgeService.upload_document(ACTOR, kb_id, old["documentId"], "test.md", b"Milvus NEW VERSION", "new-version", {})
    new_job = worker._claim_one()
    worker._run_ingest(new_job)
    worker._run_ingest(old_job)
    with db.engine.connect() as conn:
        active = conn.execute(select(document_table.c.active_build_id)).scalar_one()
    assert active == new["buildId"], {"active": active, "old": old["buildId"], "new": new["buildId"]}


def test_agent_does_not_reuse_previous_segment(monkeypatch, tmp_path):
    ingested(monkeypatch, tmp_path)
    conv = QaService.create_conversation(ACTOR, {})
    submit_test_run(ACTOR, conv["conversationId"], {"question": "Milvus"})
    QaService.create_segment(ACTOR, conv["conversationId"], {"scope": {"mode": "ALL", "kbIds": []}})
    import app.qa.service as qa
    monkeypatch.setattr(qa, "is_test_env", lambda: False)
    current = submit_test_run(ACTOR, conv["conversationId"], {"question": "New segment", "mode": "AGENT"})
    messages = LangGraphAgent._conversation_memory(ACTOR, conv["conversationId"], current["runId"])
    assert messages == [], messages


def test_regular_rag_followup_receives_history(monkeypatch, tmp_path):
    # 新建问答统一调用 chat_tools；原审计测试监听 chat 已不再覆盖生产路径。
    ingested(monkeypatch, tmp_path)
    conv = QaService.create_conversation(ACTOR, {})
    first = submit_test_run(ACTOR, conv["conversationId"], {"question": "Milvus first question"})
    assert QaService.get_run(ACTOR, first["runId"])["status"] in {"SUCCEEDED", "COMPLETED", "NO_EVIDENCE"}
    observed = []
    real_chat_tools = ModelGateway.chat_tools

    def capture(**kwargs):
        response = real_chat_tools(**kwargs)
        observed.append({"messages": kwargs["messages"], "reply": response})
        return response

    monkeypatch.setattr(ModelGateway, "chat_tools", capture)
    current_question = "Milvus: what about its limitations?"
    second = submit_test_run(ACTOR, conv["conversationId"], {"question": current_question})
    assert QaService.get_run(ACTOR, second["runId"])["status"] in {"SUCCEEDED", "COMPLETED", "NO_EVIDENCE"}
    assert any(
        any(m.get("role") == "user" and m.get("content") == "Milvus first question"
            for m in call["messages"])
        and any(m.get("role") == "user" and m.get("content") == current_question
                for m in call["messages"])
        for call in observed
    ), observed
    assert any(
        tc.get("name") == "search_knowledge" and tc.get("args", {}).get("query") == current_question
        for call in observed for tc in call["reply"].get("tool_calls", [])
    ), observed


def test_production_upsert_does_not_fill_mock_store(monkeypatch):
    import app.retrieval.milvus_service as milvus
    from types import SimpleNamespace
    monkeypatch.setattr(milvus, "is_test_env", lambda: False)
    monkeypatch.setattr(milvus, "get_embeddings", lambda texts, **kw: [[1.0] * 1024 for _ in texts])
    monkeypatch.setattr(MilvusHybridService, "get_client", lambda: SimpleNamespace(upsert=lambda **kw: None))
    monkeypatch.setattr(MilvusHybridService, "_mock_store", {})
    MilvusHybridService.upsert_chunks("t", "b", [{"id": "c", "text_content": "synthetic"}], "k", "d", "v")
    assert not MilvusHybridService._mock_store


def test_milvus_reconnects_after_initial_failure(monkeypatch):
    import app.retrieval.milvus_service as milvus
    import pymilvus
    from types import SimpleNamespace
    monkeypatch.setattr(milvus, "is_test_env", lambda: False)
    monkeypatch.setattr(MilvusHybridService, "_client_checked", False)
    monkeypatch.setattr(MilvusHybridService, "_client", None)
    attempts = []
    def connect(**kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("synthetic startup outage")
        return SimpleNamespace(has_collection=lambda name: True)
    monkeypatch.setattr(pymilvus, "MilvusClient", connect)
    assert MilvusHybridService.get_client() is None
    assert MilvusHybridService.get_client() is not None


def test_ui_config_post_options_are_not_silently_ignored():
    cfg = RetrievalConfig.from_payload("audit", {
        "retrieval": {"mode": "HYBRID"},
        "post": {"rerank": True, "parent": False, "mmr": True},
        "query": {"rewrite": True, "hyde": True, "multiQuery": True},
    })
    assert cfg.rerank_enabled and not cfg.parent_expansion_enabled, cfg


def test_remote_advanced_strategies_remain_supported():
    from app.retrieval.config import validate_retrieval_payload
    cfg = validate_retrieval_payload({"query": {"rewrite": True, "hyde": True, "multiQuery": True},
                                      "post": {"mmr": True}})
    assert cfg.rewrite_enabled and cfg.hyde_enabled and cfg.multi_query_enabled and cfg.mmr_enabled


def queued_qa(monkeypatch, scope=None, job_id=None):
    import app.qa.service as qa
    monkeypatch.setattr(qa, "is_test_env", lambda: False)
    conv = QaService.create_conversation(ACTOR, {})
    return submit_test_run(ACTOR, conv["conversationId"], {"question": "Milvus"}, evaluation_scope=scope, trusted_evaluation_job_id=job_id)["runId"]


def test_external_retries_each_record_usage_and_stop_before_exceeding_budget(monkeypatch):
    from app.qa.model_usage import invoke_tracked
    from app.qa.runtime import retry_call, AgentRuntimePolicy
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "2")
    monkeypatch.setenv("RAG_MONTHLY_BUDGET_CNY", "100")
    set_prices()
    run_id = queued_qa(monkeypatch)
    calls = []
    def request():
        calls.append(1)
        if len(calls) == 1:
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", "synthetic outage")
        return {"input_tokens": 5, "output_tokens": 3}
    def invoke():
        return invoke_tracked(request, tenant_id=ACTOR.tenant_id, run_id=run_id,
            purpose="ANSWER", alias="chat-main", payload="synthetic", max_output_tokens=10,
            usage_reader=lambda result: result)
    retry_call("audit-budget", invoke, policy=AgentRuntimePolicy(retry_base_delay_seconds=0.001))
    with pytest.raises(ApiException) as exc:
        invoke()
    assert exc.value.code == "BUDGET_EXCEEDED"
    assert len(calls) == 2
    with db.engine.connect() as conn:
        rows = conn.execute(select(model_call_table).where(model_call_table.c.run_id == run_id).order_by(model_call_table.c.created_at)).all()
    assert [r.status for r in rows] == ["UNKNOWN", "SETTLED"]
    assert len({r.id for r in rows}) == 2
    assert rows[1].input_tokens == 5 and rows[1].output_tokens == 3
    assert rows[0].settled_amount is None and rows[1].settled_amount is not None


def test_batch_reservation_shared_by_multiple_qa_runs(monkeypatch):
    from app.qa.model_usage import invoke_tracked
    from app.models import job_table
    set_prices()
    monkeypatch.setenv("RAG_MONTHLY_BUDGET_CNY", "100")
    with db.engine.begin() as conn:
        conn.execute(job_table.insert().values(tenant_id=ACTOR.tenant_id, id="audit-eval-job",
            kind="EVALUATE", target_id="audit-eval", requested_by=ACTOR.user_id,
            idempotency_key="audit-eval-job", request_hash=b"audit", status="RUNNING",
            stage="EVALUATING", checkpoint={"maxCostCny": 0.0004, "maxModelCalls": 100}))
    first = queued_qa(monkeypatch, job_id="audit-eval-job")
    actual = []
    def invoke(run_id):
        return invoke_tracked(lambda: actual.append(run_id), tenant_id=ACTOR.tenant_id,
            run_id=run_id, purpose="ANSWER", alias="chat-main", payload="x")
    invoke(first)
    with db.engine.begin() as conn:
        conn.execute(update(qa_run_table).where(qa_run_table.c.id == first).values(status="SUCCEEDED"))
    second = queued_qa(monkeypatch, job_id="audit-eval-job")
    with pytest.raises(ApiException) as exc:
        invoke(second)
    assert exc.value.code == "BUDGET_EXCEEDED"
    assert actual == [first]


def test_unknown_prices_fail_before_external_call(monkeypatch):
    from app.qa.model_usage import invoke_tracked
    sent = []
    with pytest.raises(ApiException) as exc:
        invoke_tracked(lambda: sent.append(1), tenant_id=ACTOR.tenant_id,
            purpose="ANSWER", alias="unconfigured-model", payload="test")
    assert exc.value.code == "MODEL_PRICING_NOT_CONFIGURED" and sent == []


def test_embedding_batches_belong_to_the_run(monkeypatch):
    from app.retrieval import embedding
    run_id = queued_qa(monkeypatch)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://synthetic/v1")
    class Response:
        status_code = 200
        def json(self):
            return {"data": [{"index": 0, "embedding": [1.0] * 4}], "usage": {"total_tokens": 3}}
    class Client:
        def post(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(embedding, "get_http_client", Client)
    assert len(embedding.get_embeddings(["a", "b"], dim=4, batch_size=1, tenant_id=ACTOR.tenant_id, run_id=run_id)) == 2
    with db.engine.connect() as conn:
        calls = conn.execute(select(model_call_table).where(model_call_table.c.run_id == run_id)).all()
    assert len(calls) == 2 and all(c.purpose == "EMBEDDING" and c.input_tokens == 3 for c in calls)


def test_eval_persona_missing_or_disabled_never_falls_back():
    from app.evaluation.runner import EvaluationRunner
    for login in ("missing", "disabled_user"):
        with pytest.raises(ApiException) as exc:
            EvaluationRunner._persona(ACTOR, {"persona": {"loginName": login}})
        assert exc.value.code == "PERSONA_UNAVAILABLE"


def test_eval_multiturn_uses_same_conversation_and_snapshot(monkeypatch):
    from app.evaluation.runner import EvaluationRunner
    from app.evaluation.catalog import EvaluationCatalog
    submitted = []
    scope = {"snapshotId": "snap", "evalRunId": "eval", "documents": [{"documentId": "d", "buildId": "b"}]}
    monkeypatch.setattr(EvaluationCatalog, "check_snapshot", lambda *args, **kwargs: ["d"])
    monkeypatch.setattr(QaService, "create_conversation", lambda *args: {"conversationId": "one-conversation"})
    monkeypatch.setattr(QaService, "get_conversation_detail", lambda *args: {"currentSegmentId": "segment"})
    def submit(actor, conv, body, **kw):
        submitted.append((actor.user_id, conv, body, kw))
        return {"runId": str(len(submitted))}
    monkeypatch.setattr(QaService, "submit_run", submit)
    monkeypatch.setattr(QaService, "get_run", lambda *args: {"status": "COMPLETED"})
    _, ids = EvaluationRunner._run_qa_turns(ACTOR, ACTOR, "eval", "result",
        {"question": {"turns": [{"text": "first"}, {"text": "followup"}]}},
        {"mode": "RAG", "configId": "cfg"}, scope)
    assert ids == ["1", "2"]
    assert [s[2]["question"] for s in submitted] == ["first", "followup"]
    assert all(s[1] == "one-conversation" and s[3]["evaluation_scope"] == scope for s in submitted)


def test_snapshot_scope_limits_real_retrieval_and_rejects_changed_build(monkeypatch, tmp_path):
    from app.retrieval.pipeline import RetrievalPipeline
    first = ingested(monkeypatch, tmp_path)
    with db.engine.connect() as conn:
        kb_id = conn.execute(select(document_table.c.kb_id)).scalar_one()
    second = KnowledgeService.upload_document(ACTOR, kb_id, None, "second.md", b"Milvus outside snapshot", "second", {})
    scope = {"documents": [{"documentId": first["documentId"], "buildId": first["buildId"]}]}
    run_id = queued_qa(monkeypatch, scope)
    hits = RetrievalPipeline.retrieve(ACTOR, "Milvus", run_id=run_id)["hits"]
    assert hits and {h["documentId"] for h in hits} == {first["documentId"]}
    with db.engine.begin() as conn:
        conn.execute(update(document_table).where(document_table.c.id == first["documentId"]).values(active_build_id="changed"))
    with pytest.raises(ApiException) as exc:
        RetrievalPipeline.retrieve(ACTOR, "Milvus", run_id=run_id)
    assert exc.value.code == "SOURCE_CHANGED"


def test_no_late_publish_after_trash_restore_epoch(monkeypatch, tmp_path):
    doc = previous.queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    job = worker._claim_one()
    with db.engine.begin() as conn:
        conn.execute(update(document_table).values(lifecycle_epoch=2))
    worker._run_ingest(job)
    with db.engine.connect() as conn:
        assert conn.execute(select(document_table.c.active_build_id)).scalar_one() is None


def test_eval_restart_reuses_existing_turns(monkeypatch, tmp_path):
    from app.evaluation.runner import EvaluationRunner
    from app.evaluation.catalog import EvaluationCatalog
    ingested(monkeypatch, tmp_path)
    monkeypatch.setattr(EvaluationCatalog, "check_snapshot", lambda *args, **kwargs: None)
    case = {"question": {"turns": [{"text": "Milvus first"}, {"text": "Milvus second"}]}}
    manifest = {"mode": "RAG", "configId": "cfg-default-retrieval"}
    with db.engine.connect() as conn:
        doc = conn.execute(select(document_table)).first()
    scope = {"snapshotId": "snap", "evalRunId": "eval",
             "documents": [{"documentId": doc.id, "buildId": doc.active_build_id}]}
    _, first = EvaluationRunner._run_qa_turns(ACTOR, ACTOR, "eval", "res", case, manifest, scope)
    def unexpected(*args):
        pytest.fail("恢复评测不能创建新会话")
    monkeypatch.setattr(QaService, "create_conversation", unexpected)
    _, resumed = EvaluationRunner._run_qa_turns(ACTOR, ACTOR, "eval", "res", case, manifest, scope)
    assert resumed == first


def test_gateway_fallback_has_separate_reservation_and_actual_alias(monkeypatch):
    from types import SimpleNamespace
    set_prices()
    set_prices("judge-main")
    run_id = queued_qa(monkeypatch)
    sent = []
    class Primary:
        def bind(self, **kwargs):
            assert kwargs == {"response_format": {"type": "json_object"}}
            return self
        def invoke(self, messages):
            sent.append("chat-main")
            raise RuntimeError("synthetic provider failure")
    class Fallback:
        def bind(self, **kwargs):
            assert kwargs == {"response_format": {"type": "json_object"}}
            return self
        def invoke(self, messages):
            sent.append("judge-main")
            return SimpleNamespace(content="answer", usage_metadata={"input_tokens": 5, "output_tokens": 2})
    monkeypatch.setenv("LITELLM_BASE_URL", "http://synthetic/v1")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL_ALIAS", "judge-main")
    monkeypatch.setattr("app.qa.model_gateway.get_chat_model",
                        lambda alias, *args: Primary() if alias == "chat-main" else Fallback())
    assert ModelGateway.chat(ACTOR.tenant_id, run_id, "ANSWER", [{"role": "user", "content": "test"}]) == "answer"
    with db.engine.connect() as conn:
        rows = conn.execute(select(model_call_table).where(model_call_table.c.run_id == run_id)
                            .order_by(model_call_table.c.created_at)).all()
    assert sent == ["chat-main", "judge-main"]
    assert [r.status for r in rows] == ["UNKNOWN", "SETTLED"]
    assert [r.model_config_id for r in rows] == ["cfg-model-chat-main", "cfg-model-judge-main"]
    assert len({r.reservation_id for r in rows}) == 2
