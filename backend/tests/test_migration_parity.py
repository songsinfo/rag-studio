import json
import io
import uuid
import zipfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select, update

import app.db as app_db
from app.errors import ApiException
from app.identity.deps import Actor
from app.knowledge.service import KnowledgeService
from app.knowledge.chunker import adapt_docling_hybrid_chunks
from app.knowledge.imports import create_import_preview, consume_import_preview, split_wiki_package
from app.model_budget import BudgetService
from app.models import (
    document_table,
    artifact_table,
    index_build_table,
    eval_case_table,
    eval_dataset_version_table,
    eval_gold_table,
    eval_snapshot_document_table,
    eval_snapshot_table,
    chunk_table,
    model_call_table,
    job_table,
    budget_reservation_table,
    config_revision_table,
    qa_run_table,
    run_source_table,
    document_grant_table,
)
from app.qa.service import QaService
from app.qa.model_gateway import ModelGateway
from app.retrieval.config import RetrievalConfig
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.milvus_service import MilvusHybridService


ACTOR = Actor(
    tenant_id="tenant-demo-1",
    user_id="user-admin-1",
    admin=True,
    security_epoch=1,
)


def test_strict_judge_requires_complete_claim_coverage():
    claims = [{"id": "c1"}, {"id": "c2"}]

    with pytest.raises(ApiException) as exc:
        QaService._validate_strict_verdicts(claims, {"verdicts": []})
    assert exc.value.code == "VERIFICATION_FAILED"

    with pytest.raises(ApiException):
        QaService._validate_strict_verdicts(
            claims,
            {"verdicts": [{"claimId": "c1", "status": "SUPPORTED"}]},
        )

    QaService._validate_strict_verdicts(
        claims,
        {
            "verdicts": [
                {"claimId": "c1", "status": "SUPPORTED"},
                {"claimId": "c2", "status": "SUPPORTED"},
            ]
        },
    )


def test_client_cannot_override_retrieval_config():
    conv = QaService.create_conversation(
        ACTOR,
        {"title": "config parity", "scope": {"mode": "ALL"}},
    )
    with pytest.raises(ApiException) as exc:
        QaService.submit_run(
            ACTOR,
            conv["conversationId"],
            {
                "question": "系统如何检索？",
                "segmentId": conv["currentSegmentId"],
                "mode": "RAG",
                "configId": "cfg-attacker-selected",
            },
            idempotency_key="config-override",
        )
    assert exc.value.code == "INVALID_INPUT"


def test_upload_idempotency_hash_includes_file_content():
    kb = KnowledgeService.create_kb(ACTOR, "Idempotency KB", [])
    kb_id = kb["id"]

    KnowledgeService.upload_document(
        ACTOR,
        kb_id,
        None,
        "same.txt",
        b"AA",
        "same-key",
        {"embedding": True},
    )

    with pytest.raises(ApiException) as exc:
        KnowledgeService.upload_document(
            ACTOR,
            kb_id,
            None,
            "same.txt",
            b"BB",
            "same-key",
            {"embedding": True},
        )
    assert exc.value.code == "IDEMPOTENCY_CONFLICT"


def test_parent_over_budget_falls_back_to_child():
    now = datetime.now(timezone.utc)
    build_id = "build-parent-budget"
    parent_id = "parent-budget"
    child_id = "child-budget"
    with app_db.engine.begin() as conn:
        conn.execute(
            chunk_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=parent_id,
                build_id=build_id,
                parent_id=None,
                kind="PARENT",
                ordinal=0,
                text_content="P" * 100,
                token_count=100,
                content_hash=b"p" * 32,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            chunk_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=child_id,
                build_id=build_id,
                parent_id=parent_id,
                kind="CHILD",
                ordinal=0,
                text_content="child evidence",
                token_count=3,
                content_hash=b"c" * 32,
                created_at=now,
                updated_at=now,
            )
        )

    warnings = []
    result = RetrievalPipeline._expand_parents(
        ACTOR.tenant_id,
        [
            {
                "chunkId": child_id,
                "buildId": build_id,
                "text": "child evidence",
                "score": 1.0,
            }
        ],
        RetrievalConfig(config_id="test", parent_max_chars=10),
        warnings,
    )
    assert result[0]["text"] == "child evidence"
    assert result[0].get("parentChunkId") is None
    assert "PARENT_TRIMMED" in warnings


def test_purge_scrubs_evaluation_derivatives():
    kb = KnowledgeService.create_kb(ACTOR, "Purge Eval KB", [])
    uploaded = KnowledgeService.upload_document(
        ACTOR,
        kb["id"],
        None,
        "secret.txt",
        b"secret evaluation source",
        "purge-eval-upload",
        {"embedding": True},
    )
    doc_id = uploaded["documentId"]
    now = datetime.now(timezone.utc)

    dataset_id = "dataset-purge"
    case_id = "case-purge"
    snapshot_id = "snapshot-purge"

    with app_db.engine.begin() as conn:
        conn.execute(
            eval_dataset_version_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=dataset_id,
                name="purge",
                revision=1,
                status="FROZEN",
                manifest_hash=b"d" * 32,
                frozen_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            eval_case_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=case_id,
                dataset_version_id=dataset_id,
                case_key="purge-case",
                split="TEST",
                group_key="purge",
                category="FACTUAL",
                question_json={"question": "secret?"},
                persona_json={},
                expected_json={"evidenceGroups": [{"quote": "secret"}]},
                annotation_status="HUMAN_REVIEWED",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            eval_gold_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=str(uuid.uuid4()),
                case_id=case_id,
                fact_group="secret",
                alternative_no=0,
                origin_system="test",
                origin_tenant_id=ACTOR.tenant_id,
                origin_document_id=doc_id,
                version_hash=b"v" * 32,
                anchor_json={"quote": "secret"},
                relevance_grade=3,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            eval_snapshot_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=snapshot_id,
                name="purge",
                manifest_hash=b"s" * 32,
                manifest_json={"secret": "value"},
                status="READY",
                environment_id="test",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            eval_snapshot_document_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=str(uuid.uuid4()),
                snapshot_id=snapshot_id,
                document_id=doc_id,
                version_id=uploaded["versionId"],
                origin_system="test",
                origin_tenant_id=ACTOR.tenant_id,
                origin_document_id=doc_id,
                origin_version_hash=b"v" * 32,
                created_at=now,
                updated_at=now,
            )
        )

    with app_db.engine.connect() as conn:
        row_version = conn.execute(
            select(document_table.c.row_version).where(document_table.c.id == doc_id)
        ).scalar_one()
    KnowledgeService.document_lifecycle(ACTOR, doc_id, "trash", int(row_version))

    with app_db.engine.connect() as conn:
        row_version = conn.execute(
            select(document_table.c.row_version).where(document_table.c.id == doc_id)
        ).scalar_one()
    KnowledgeService.document_lifecycle(ACTOR, doc_id, "purge", int(row_version))

    with app_db.engine.connect() as conn:
        gold = conn.execute(
            select(eval_gold_table.c.anchor_json).where(eval_gold_table.c.case_id == case_id)
        ).scalar_one()
        case = conn.execute(
            select(
                eval_case_table.c.question_json,
                eval_case_table.c.expected_json,
                eval_case_table.c.annotation_status,
            ).where(eval_case_table.c.id == case_id)
        ).one()
        dataset_status = conn.execute(
            select(eval_dataset_version_table.c.status).where(
                eval_dataset_version_table.c.id == dataset_id
            )
        ).scalar_one()
        snapshot = conn.execute(
            select(eval_snapshot_table.c.status, eval_snapshot_table.c.manifest_json).where(
                eval_snapshot_table.c.id == snapshot_id
            )
        ).one()

    assert gold == {}
    assert case.question_json == {}
    assert case.expected_json == {}
    assert case.annotation_status == "PURGED"
    assert dataset_status == "PURGED"
    assert snapshot.status == "PURGED"
    assert snapshot.manifest_json == {}



def test_directory_import_rechecks_source_hash(tmp_path, monkeypatch):
    root = tmp_path / "import-root"
    root.mkdir()
    source = root / "guide.md"
    source.write_text("# Guide\noriginal", encoding="utf-8")
    monkeypatch.setenv("RAG_IMPORT_ROOT", str(root))

    preview = create_import_preview(ACTOR.tenant_id, ACTOR.user_id, ".")
    assert preview["files"][0]["relativePath"] == "guide.md"

    source.write_text("# Guide\nchanged", encoding="utf-8")
    with pytest.raises(ApiException) as exc:
        consume_import_preview(
            ACTOR.tenant_id,
            ACTOR.user_id,
            preview["estimateId"],
            ["guide.md"],
        )
    assert exc.value.code == "SOURCE_CHANGED"


def test_wiki_import_rejects_traversal_and_remote_assets():
    def make_zip(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        return buffer.getvalue()

    with pytest.raises(ApiException) as exc:
        split_wiki_package(make_zip({"../escape.md": "# bad"}))
    assert exc.value.code == "INVALID_ARCHIVE"

    with pytest.raises(ApiException) as exc:
        split_wiki_package(
            make_zip({"page.md": "![remote](https://example.invalid/image.png)"})
        )
    assert exc.value.code == "INVALID_ARCHIVE"

    pages = split_wiki_package(
        make_zip(
            {
                "page.md": "# Page\n![local](assets/p.png)",
                "assets/p.png": b"\x89PNG\r\n\x1a\n",
            }
        )
    )
    assert len(pages) == 1
    assert b"data:image/png;base64," in pages[0]["bytes"]


def test_not_sent_budget_reservation_is_released():
    call_id = str(uuid.uuid4())
    reserved = BudgetService.reserve(
        tenant_id=ACTOR.tenant_id,
        call_id=call_id,
        model_alias="chat-main",
        input_upper_bound=100,
        output_upper_bound=100,
    )
    BudgetService.settle(
        ACTOR.tenant_id,
        reserved["id"],
        "NOT_SENT",
        Decimal("0"),
    )
    summary = BudgetService.summary(ACTOR.tenant_id)
    assert summary
    assert Decimal(summary[0]["reservedAmount"]) == Decimal("0")
    assert Decimal(summary[0]["settledAmount"]) == Decimal("0")



def test_docling_warning_metadata_survives_adapter():
    units, _, _ = adapt_docling_hybrid_chunks(
        [
            {
                "chunk_index": 0,
                "text": "warning text",
                "raw_text": "warning text",
                "num_tokens": 3,
                "headings": ["Section"],
                "metadata": {
                    "doclingWarnings": [
                        {"code": "LOW_OCR_CONFIDENCE", "detail": "page 1"}
                    ]
                },
            }
        ]
    )
    assert units
    assert units[0]["location"]["metadata"]["doclingWarnings"][0]["code"] == "LOW_OCR_CONFIDENCE"


def test_legacy_java_retrieval_config_maps_to_python_execution_fields():
    cfg = RetrievalConfig.from_payload(
        "legacy",
        {
            "query": {
                "rewrite": True,
                "multiQuery": True,
                "hyde": True,
                "maxVariants": 3,
            },
            "retrieval": {
                "mode": "HYBRID",
                "perRouteK": 40,
                "numCandidates": 300,
                "representations": ["ORIGINAL", "SUMMARY"],
            },
            "fusion": {"rrfK": 75},
            "post": {
                "rerank": True,
                "rerankTopN": 20,
                "allowRerankFallback": False,
                "mmr": True,
                "parent": True,
                "lambda": 0.6,
            },
            "context": {"maxTokens": 5000, "maxUnits": 6},
        },
    )
    assert cfg.dense_top_k == 40
    assert cfg.sparse_top_k == 40
    assert cfg.num_candidates == 300
    assert cfg.rrf_k == 75
    assert cfg.max_variants == 3
    assert cfg.rerank_enabled is True
    assert cfg.rerank_top_k == 20
    assert cfg.rerank_fail_open is False
    assert cfg.final_top_k == 6
    assert cfg.context_max_bytes == 5000
    assert cfg.parent_max_chars == 5000
    assert cfg.representations == ("ORIGINAL", "SUMMARY")


def test_model_call_can_be_owned_by_job():
    call_id = str(uuid.uuid4())
    ModelGateway._record_call(
        tenant_id=ACTOR.tenant_id,
        call_id=call_id,
        run_id=None,
        purpose="EMBEDDING",
        status="SUCCESS",
        input_tokens=10,
        output_tokens=0,
        model_alias="embedding-main",
        job_id="job-test-owner",
    )
    with app_db.engine.connect() as conn:
        row = conn.execute(
            select(
                model_call_table.c.run_id,
                model_call_table.c.job_id,
            ).where(model_call_table.c.id == call_id)
        ).one()
    assert row.run_id is None
    assert row.job_id == "job-test-owner"


def test_latest_published_model_pricing_overrides_seed_zero_price():
    now = datetime.now(timezone.utc)
    cfg_id = str(uuid.uuid4())
    with app_db.engine.begin() as conn:
        conn.execute(
            config_revision_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=cfg_id,
                kind="MODEL",
                name="chat-main-real-price",
                revision=99,
                sha256=b"p" * 32,
                payload={
                    "modelAlias": "chat-main",
                    "currency": "CNY",
                    "inputPricePerMillion": 2.5,
                    "outputPricePerMillion": 10,
                },
                status="PUBLISHED",
                created_at=now,
                updated_at=now,
            )
        )
    pricing = BudgetService.pricing(ACTOR.tenant_id, "chat-main")
    assert pricing["configId"] == cfg_id
    assert pricing["inputPricePerMillion"] == Decimal("2.5")
    assert pricing["outputPricePerMillion"] == Decimal("10")


def test_peer_deletion_requires_ack(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"acknowledged": True, "copiesPurged": 1}

    class Client:
        @staticmethod
        def post(*args, **kwargs):
            return Response()

    monkeypatch.setenv("RAG_DELETION_PEERS", "https://eval.internal")
    monkeypatch.setenv("RAG_DELETION_API_TOKEN", "x" * 32)
    monkeypatch.setattr("app.knowledge.service.get_http_client", lambda: Client())

    domains = KnowledgeService._propagate_deletion_to_peers(
        "demo",
        ACTOR.tenant_id,
        "doc-1",
    )
    assert domains["peer:https://eval.internal"] == "COMPLETE"



def test_model_send_gate_rejects_revoked_history_source():
    kb = KnowledgeService.create_kb(ACTOR, "History ACL KB", [])
    uploaded = KnowledgeService.upload_document(
        ACTOR,
        kb["id"],
        None,
        "history.txt",
        b"history secret",
        "history-acl-upload",
        {"embedding": True},
    )
    conv = QaService.create_conversation(ACTOR, {"scope": {"mode": "ALL"}})
    run = QaService.submit_run(
        ACTOR,
        conv["conversationId"],
        {"question": "first", "segmentId": conv["currentSegmentId"], "mode": "RAG"},
        idempotency_key="history-acl-run",
    )
    now = datetime.now(timezone.utc)
    with app_db.engine.begin() as conn:
        conn.execute(
            qa_run_table.update()
            .where(qa_run_table.c.id == run["runId"])
            .values(status="RUNNING")
        )
        conn.execute(
            run_source_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=str(uuid.uuid4()),
                run_id=run["runId"],
                document_id=uploaded["documentId"],
                build_id=uploaded["buildId"],
                unit_id=None,
                evidence_id=None,
                purpose="HISTORY",
                span_json=None,
                payload_hash=None,
                send_status="PREPARED",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_grant_table.delete().where(
                document_grant_table.c.document_id == uploaded["documentId"]
            )
        )

    with pytest.raises(ApiException) as exc:
        ModelGateway.assert_run_sources_current(ACTOR.tenant_id, run["runId"])
    assert exc.value.code == "SOURCE_CHANGED"


def test_real_milvus_hybrid_fuses_after_canonical_dedupe(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def search(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [[
                    {"distance": 0.99, "entity": {"chunk_id": "c1", "kb_id": "k", "document_id": "d", "version_id": "v", "build_id": "b", "index_epoch": 1, "representation_type": "ORIGINAL", "text_content": "c1 original"}},
                    {"distance": 0.98, "entity": {"chunk_id": "c1", "kb_id": "k", "document_id": "d", "version_id": "v", "build_id": "b", "index_epoch": 1, "representation_type": "SUMMARY", "text_content": "c1 summary"}},
                    {"distance": 0.97, "entity": {"chunk_id": "c2", "kb_id": "k", "document_id": "d", "version_id": "v", "build_id": "b", "index_epoch": 1, "representation_type": "ORIGINAL", "text_content": "c2"}},
                ]]
            return [[
                {"distance": 10.0, "entity": {"chunk_id": "c2", "kb_id": "k", "document_id": "d", "version_id": "v", "build_id": "b", "index_epoch": 1, "representation_type": "QUESTION", "text_content": "c2 question"}},
                {"distance": 9.0, "entity": {"chunk_id": "c3", "kb_id": "k", "document_id": "d", "version_id": "v", "build_id": "b", "index_epoch": 1, "representation_type": "ORIGINAL", "text_content": "c3"}},
            ]]

    fake = FakeClient()
    monkeypatch.setattr(MilvusHybridService, "get_client", classmethod(lambda cls: fake))
    monkeypatch.setattr(
        "app.retrieval.milvus_service.get_embeddings",
        lambda *args, **kwargs: [[1.0] * 1024],
    )

    hits = MilvusHybridService.search(
        tenant_id=ACTOR.tenant_id,
        query="q",
        allowed_build_ids=["b"],
        dense_top_k=3,
        sparse_top_k=3,
        fusion_top_k=3,
        mode="HYBRID",
        rrf_enabled=True,
        rrf_k=60,
        allowed_build_epochs={"b": 1},
        representations=["ORIGINAL", "SUMMARY", "QUESTION"],
        run_id="run-test",
        num_candidates=20,
    )
    ids = [hit["chunkId"] for hit in hits]
    assert len(ids) == len(set(ids))
    assert ids[0] == "c2"
    assert set(ids) == {"c1", "c2", "c3"}


def test_rewrite_keeps_original_and_rewritten_query(monkeypatch):
    monkeypatch.setattr(
        ModelGateway,
        "chat",
        classmethod(lambda cls, *args, **kwargs: json.dumps({"query": "独立改写问题"}, ensure_ascii=False)),
    )
    warnings = []
    independent, variants, hyde = RetrievalPipeline._prepare_queries(
        actor=ACTOR,
        query="原始追问",
        history=[{"role": "user", "content": "上一轮问题"}],
        cfg=RetrievalConfig(
            config_id="rewrite-parity",
            rewrite_enabled=True,
            max_variants=4,
        ),
        run_id="run-rewrite-parity",
        job_id=None,
        warnings=warnings,
    )
    assert independent == "独立改写问题"
    assert variants == ["原始追问", "独立改写问题"]
    assert hyde is None
    assert warnings == []


def test_model_gateway_rejects_cancelled_background_job():
    now = datetime.now(timezone.utc)
    job_id = "job-model-gate"
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=job_id,
                kind="INGEST",
                target_id="doc-model-gate",
                requested_by=ACTOR.user_id,
                idempotency_key="job-model-gate-key",
                request_hash=b"job-model-gate",
                status="RUNNING",
                stage="INDEXING",
                attempt=0,
                available_at=now,
                lease_epoch=1,
                created_at=now,
                updated_at=now,
            )
        )
    ModelGateway.assert_job_current(ACTOR.tenant_id, job_id)
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.update()
            .where(job_table.c.id == job_id)
            .values(status="CANCELLED", cancel_requested_at=now)
        )
    with pytest.raises(ApiException) as exc:
        ModelGateway.assert_job_current(ACTOR.tenant_id, job_id)
    assert exc.value.code == "CANCELLED"


def test_direct_evaluation_job_reservation_uses_batch_owner():
    now = datetime.now(timezone.utc)
    job_id = "job-eval-budget-owner"
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=job_id,
                kind="EVALUATE",
                target_id="eval-budget-owner",
                requested_by=ACTOR.user_id,
                idempotency_key="eval-budget-owner-key",
                request_hash=b"eval-budget-owner",
                status="RUNNING",
                stage="EVALUATING",
                attempt=0,
                available_at=now,
                lease_epoch=1,
                checkpoint={"maxCostCny": 10, "maxModelCalls": 100},
                created_at=now,
                updated_at=now,
            )
        )
    result = BudgetService.reserve(
        ACTOR.tenant_id,
        "call-eval-budget-owner",
        "chat-main",
        100,
        100,
        job_id=job_id,
    )
    assert result["status"] == "RESERVED"
    with app_db.engine.begin() as conn:
        row = conn.execute(
            select(budget_reservation_table.c.caller_environment).where(
                budget_reservation_table.c.caller_call_id == "call-eval-budget-owner"
            )
        ).first()
    assert row is not None
    assert str(row.caller_environment).endswith("|eval:" + job_id)


def test_fallback_model_rechecks_background_job_gate(monkeypatch):
    now = datetime.now(timezone.utc)
    job_id = "job-fallback-gate"
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=job_id,
                kind="INGEST",
                target_id="doc-fallback-gate",
                requested_by=ACTOR.user_id,
                idempotency_key="job-fallback-gate-key",
                request_hash=b"job-fallback-gate",
                status="RUNNING",
                stage="INDEXING",
                attempt=0,
                available_at=now,
                lease_epoch=1,
                created_at=now,
                updated_at=now,
            )
        )

    class Primary:
        def invoke(self, messages):
            with app_db.engine.begin() as conn:
                conn.execute(
                    job_table.update()
                    .where(job_table.c.id == job_id)
                    .values(status="CANCELLED", cancel_requested_at=now)
                )
            raise RuntimeError("primary failed")

    class Fallback:
        def invoke(self, messages):
            raise AssertionError("fallback must not run after job cancellation")

    def fake_model(alias, *args, **kwargs):
        return Primary() if alias == "chat-main" else Fallback()

    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.invalid")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL_ALIAS", "chat-fallback")
    monkeypatch.setattr("app.qa.model_gateway.get_chat_model", fake_model)
    monkeypatch.setattr(
        BudgetService,
        "reserve",
        classmethod(lambda cls, *args, **kwargs: {
            "id": "reservation-fallback",
            "amount": Decimal("0"),
            "pricing": {
                "configId": "cfg-model-chat-main",
                "inputPricePerMillion": Decimal("0"),
                "outputPricePerMillion": Decimal("0"),
            },
            "status": "RESERVED",
        }),
    )
    monkeypatch.setattr(
        BudgetService,
        "settle_call",
        classmethod(lambda cls, *args, **kwargs: {
            "reservedAmount": Decimal("0"),
            "settledAmount": None,
            "pricing": {"configId": "cfg-model-chat-main"},
            "reservationId": "reservation-fallback",
            "authority": "LOCAL_MONTHLY_BUDGET",
        }),
    )

    with pytest.raises(ApiException) as exc:
        ModelGateway.chat(
            ACTOR.tenant_id,
            run_id=None,
            purpose="ENRICH",
            messages=[{"role": "user", "content": "test"}],
            model_alias="chat-main",
            job_id=job_id,
        )
    assert exc.value.code == "CANCELLED"


def test_pre_dispatch_job_gate_releases_as_not_sent(monkeypatch):
    now = datetime.now(timezone.utc)
    job_id = "job-pre-dispatch-gate"
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=job_id,
                kind="INGEST",
                target_id="doc-pre-dispatch",
                requested_by=ACTOR.user_id,
                idempotency_key="job-pre-dispatch-key",
                request_hash=b"job-pre-dispatch",
                status="CANCELLED",
                stage="INDEXING",
                attempt=0,
                available_at=now,
                lease_epoch=1,
                cancel_requested_at=now,
                created_at=now,
                updated_at=now,
            )
        )

    class NeverInvoke:
        def invoke(self, messages):
            raise AssertionError("provider must not be invoked after cancellation")

    statuses = []
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.invalid")
    monkeypatch.setattr("app.qa.model_gateway.get_chat_model", lambda *args, **kwargs: NeverInvoke())
    monkeypatch.setattr(
        BudgetService,
        "reserve",
        classmethod(lambda cls, *args, **kwargs: {
            "id": "reservation-pre-dispatch",
            "amount": Decimal("0"),
            "pricing": {
                "configId": "cfg-model-chat-main",
                "inputPricePerMillion": Decimal("0"),
                "outputPricePerMillion": Decimal("0"),
            },
            "status": "RESERVED",
        }),
    )
    monkeypatch.setattr(
        ModelGateway,
        "_record_call",
        classmethod(lambda cls, *args, **kwargs: statuses.append(kwargs.get("status") if "status" in kwargs else args[4])),
    )

    with pytest.raises(ApiException) as exc:
        ModelGateway.chat(
            ACTOR.tenant_id,
            run_id=None,
            purpose="ENRICH",
            messages=[{"role": "user", "content": "test"}],
            model_alias="chat-main",
            job_id=job_id,
        )
    assert exc.value.code == "CANCELLED"
    assert statuses == ["NOT_SENT"]


def test_sent_failure_persists_unknown_model_call_status():
    now = datetime.now(timezone.utc)
    job_id = "job-unknown-audit"
    call_id = "call-unknown-audit"
    with app_db.engine.begin() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=ACTOR.tenant_id,
                id=job_id,
                kind="INGEST",
                target_id="doc-unknown-audit",
                requested_by=ACTOR.user_id,
                idempotency_key="job-unknown-audit-key",
                request_hash=b"job-unknown-audit",
                status="RUNNING",
                stage="INDEXING",
                attempt=0,
                available_at=now,
                lease_epoch=1,
                created_at=now,
                updated_at=now,
            )
        )

    BudgetService.reserve(
        ACTOR.tenant_id,
        call_id,
        "chat-main",
        100,
        100,
        job_id=job_id,
    )
    ModelGateway._record_call(
        tenant_id=ACTOR.tenant_id,
        call_id=call_id,
        run_id=None,
        purpose="ENRICH",
        status="FAILED",
        input_tokens=0,
        output_tokens=0,
        model_alias="chat-main",
        job_id=job_id,
    )

    with app_db.engine.connect() as conn:
        call_status = conn.execute(
            select(model_call_table.c.status).where(model_call_table.c.id == call_id)
        ).scalar_one()
        reservation_status = conn.execute(
            select(budget_reservation_table.c.status).where(
                budget_reservation_table.c.caller_call_id == call_id
            )
        ).scalar_one()

    assert call_status == "UNKNOWN"
    assert reservation_status == "UNKNOWN"


def test_artifact_download_and_warning_acceptance_check_document_access():
    kb = KnowledgeService.create_kb(
        ACTOR, "Access checks KB", [{"type": "USER", "userId": ACTOR.user_id}]
    )
    uploaded = KnowledgeService.upload_document(
        ACTOR,
        kb["id"],
        None,
        "access-check.txt",
        b"Access check content",
        "access-check-upload",
        {},
    )
    with app_db.engine.begin() as conn:
        artifact_id = conn.execute(
            select(artifact_table.c.id).where(
                artifact_table.c.document_id == uploaded["documentId"]
            )
        ).scalar_one()
        conn.execute(
            update(index_build_table)
            .where(index_build_table.c.id == uploaded["buildId"])
            .values(status="NEEDS_REVIEW")
        )
        conn.execute(
            update(job_table)
            .where(job_table.c.id == uploaded["jobId"])
            .values(status="NEEDS_REVIEW")
        )

    assert KnowledgeService.get_artifact(
        ACTOR, uploaded["documentId"], str(artifact_id)
    ) == b"Access check content"
    assert KnowledgeService.accept_warnings(
        ACTOR, uploaded["buildId"], "Reviewed parser warning"
    ) == {"buildId": uploaded["buildId"], "status": "QUEUED"}
