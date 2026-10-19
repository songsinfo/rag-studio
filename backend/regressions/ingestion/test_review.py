"""Ingestion and access-control regression checks; no external services."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND))
spec = importlib.util.spec_from_file_location("review_fixtures", BACKEND / "tests/conftest.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
setup_test_db = fixtures.setup_test_db

from sqlalchemy import select, text, update
import app.db as db
import app.config as config
import app.knowledge.service as service
from app.identity.deps import Actor
from app.models import job_table, document_table, source_unit_table, chunk_table
from app.runtime.worker import DurableJobWorker
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.config import RetrievalConfig
from app.retrieval.milvus_service import MilvusHybridService
from app.qa.model_gateway import ModelGateway
from app.errors import ApiException

ACTOR = Actor("tenant-demo-1", "user-admin-1", True)

@pytest.fixture
def tmp_path(request):
    path = BACKEND.parent / ".runtime" / "test" / "ingestion-regression" / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path

def queue_document(monkeypatch, tmp_path, content=b"Milvus private company knowledge."):
    monkeypatch.setenv("RAG_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_ARTIFACT_STORE", "local")
    MilvusHybridService._mock_store.clear()
    kb = service.KnowledgeService.create_kb(ACTOR, "Review KB", [{"type": "ALL"}])
    with monkeypatch.context() as patch:
        patch.setattr(config, "is_test_env", lambda: False)
        result = service.KnowledgeService.upload_document(ACTOR, kb["id"], None, "test.md", content, "review", {})
    return result

def test_revoked_hits_must_not_reach_reranker(monkeypatch, tmp_path):
    doc = queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    real_search = MilvusHybridService.search
    sent = []
    def revoke_after_search(**kwargs):
        hits = real_search(**kwargs)
        with db.engine.begin() as conn:
            conn.execute(update(document_table).values(status="TRASHED"))
        return hits
    monkeypatch.setattr(MilvusHybridService, "search", revoke_after_search)
    monkeypatch.setattr(ModelGateway, "rerank", lambda **kwargs: sent.extend(kwargs["documents"]) or [{"index": 0, "relevance_score": 1.0}])
    import app.retrieval.pipeline as pipeline
    monkeypatch.setattr(pipeline, "load_retrieval_config", lambda *args: RetrievalConfig("review", rerank_enabled=True))
    with pytest.raises(ApiException):
        RetrievalPipeline.retrieve(ACTOR, "Milvus")
    assert sent == [], f"Revoked content was sent to reranker: {sent}"

def test_expired_lease_restart_is_idempotent(monkeypatch, tmp_path):
    doc = queue_document(monkeypatch, tmp_path)
    with db.engine.begin() as conn:
        # 与生产环境的唯一性约束保持一致，并模拟进程在 SourceUnit 已持久化后崩溃，
        # 此时 chunk/index 尚未完成，用来验证恢复逻辑能正确处理半成品状态。
        conn.execute(text("CREATE UNIQUE INDEX uq_unit_key ON source_unit(tenant_id, build_id, unit_key)"))
    worker = DurableJobWorker()
    job = worker._claim_one()
    build_id = str(job["checkpoint"]["buildId"])
    with db.engine.begin() as conn:
        conn.execute(source_unit_table.insert().values(
            tenant_id=ACTOR.tenant_id,
            id="partial-unit",
            build_id=build_id,
            document_id=doc["documentId"],
            unit_key="docling_chunk_0",
            kind="TEXT",
            text_content="partial stale attempt",
            location={},
            machine_generated=False,
            artifact_id=None,
            content_hash=b"x" * 32,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        conn.execute(
            update(job_table)
            .where(job_table.c.id == job["id"])
            .values(lease_until=datetime.now(timezone.utc)-timedelta(seconds=1))
        )
    recovered = worker._claim_one()
    assert recovered["lease_epoch"] == 2
    worker._run_ingest(recovered)
    with db.engine.connect() as conn:
        status = conn.execute(select(job_table.c.status)).scalar_one()
    assert status == "COMPLETED", f"Recovered job ended as {status}"

def test_old_worker_cannot_publish_after_lease_loss(monkeypatch, tmp_path):
    doc = queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    old_job = worker._claim_one()
    original_parse = service.parse_and_chunk_document
    def lose_lease(*args):
        with db.engine.begin() as conn:
            conn.execute(update(job_table).values(lease_epoch=2))
        return original_parse(*args)
    monkeypatch.setattr(service, "parse_and_chunk_document", lose_lease)
    worker._run_ingest(old_job)
    with db.engine.connect() as conn:
        active = conn.execute(select(document_table.c.active_build_id)).scalar_one()
    assert active is None, f"Stale worker published build {active}"

def test_parent_context_provenance_covers_all_units(monkeypatch, tmp_path):
    content = (("Milvus " + "A" * 410) + "\n\n" + ("Private " + "B" * 410)).encode()
    doc = queue_document(monkeypatch, tmp_path, content)
    worker = DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    with db.engine.connect() as conn:
        units = conn.execute(select(source_unit_table.c.id)).scalars().all()
        child = conn.execute(select(chunk_table).where(chunk_table.c.kind == "CHILD").order_by(chunk_table.c.ordinal)).first()
    hit = {"buildId": doc["buildId"], "chunkId": child.id, "text": child.text_content}
    expanded = RetrievalPipeline._expand_parents(ACTOR.tenant_id, [hit], RetrievalConfig("review"))
    enriched = RetrievalPipeline._attach_provenance(ACTOR.tenant_id, expanded)[0]
    assert "Private" in enriched["text"]
    assert {u["unitId"] for u in enriched["sourceUnits"]} == set(units)

def test_s3_delete_failure_must_propagate(monkeypatch):
    import app.knowledge.artifacts as artifacts
    class BrokenS3:
        def delete_object(self, **kwargs):
            raise RuntimeError("AccessDenied")
    monkeypatch.setenv("RAG_ARTIFACT_STORE", "s3")
    monkeypatch.setattr(artifacts, "_s3_client", lambda: BrokenS3())
    with pytest.raises(Exception):
        artifacts.delete_artifact("tenant/document/artifact/test.md")
