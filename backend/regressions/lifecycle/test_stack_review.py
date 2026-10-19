"""Run with REVIEW_ROOT set to the repository root; no external services required."""
import importlib.util
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(os.environ["REVIEW_ROOT"])
sys.path.insert(0, str(ROOT / "backend"))
spec = importlib.util.spec_from_file_location("previous_review", ROOT / "backend/regressions/ingestion/test_review.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
setup_test_db = previous.setup_test_db
tmp_path = previous.tmp_path
queue_document = previous.queue_document
ACTOR = previous.ACTOR

from sqlalchemy import select, text, update
import app.db as db
from app.models import document_table, index_build_table, job_table, qa_run_table, run_source_table
from app.knowledge.service import KnowledgeService
from app.runtime.worker import DurableJobWorker
from app.retrieval.milvus_service import MilvusHybridService
from app.retrieval.pipeline import RetrievalPipeline
from app.qa.service import QaService


def lifecycle(doc_id, action):
    with db.engine.connect() as conn:
        version = conn.execute(select(document_table.c.row_version).where(document_table.c.id == doc_id)).scalar_one()
    return KnowledgeService.document_lifecycle(ACTOR, doc_id, action, version)


def queued_run(monkeypatch):
    import app.qa.service as qa
    monkeypatch.setattr(qa, "is_test_env", lambda: False)
    conv = QaService.create_conversation(ACTOR, {"title": "review"})
    return QaService.submit_run(
        ACTOR,
        conv["conversationId"],
        {
            "question": "Milvus",
            "segmentId": conv["currentSegmentId"],
            "mode": "AGENT",
        },
        idempotency_key=str(uuid.uuid4()),
    )["runId"]


def test_restore_rebuild_preserves_published_epoch(monkeypatch, tmp_path):
    doc = queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    assert RetrievalPipeline.retrieve(ACTOR, "Milvus")["hits"]
    lifecycle(doc["documentId"], "trash")
    # restore 使用 upsert 是为了兼容外部索引已丢失/重建的情况：恢复时不能假设向量记录一定还存在。
    MilvusHybridService.delete_by_document(ACTOR.tenant_id, doc["documentId"])
    lifecycle(doc["documentId"], "restore")
    assert RetrievalPipeline.retrieve(ACTOR, "Milvus")["hits"], "RESTORED document has epoch-0 vectors but published fence is 1"


def test_qa_submit_satisfies_production_job_kind_constraint(monkeypatch):
    # SQLite 测试元数据不会自动带上 deploy/compose.yml 中 MySQL 的 CHECK 约束，
    # 因此用 trigger 精确模拟同一 allowlist，在不依赖 MySQL 的情况下保持测试约束一致。
    schema = (ROOT / "docs/design/schema.sql").read_text(encoding="utf-8")
    import re
    allowed = re.search(r"CHECK \(kind IN \(([^)]+)\)\)", schema[schema.index("CREATE TABLE job ("):]).group(1)
    with db.engine.begin() as conn:
        conn.execute(text(f"CREATE TRIGGER review_job_kind BEFORE INSERT ON job WHEN NEW.kind NOT IN ({allowed}) BEGIN SELECT RAISE(ABORT, 'production job kind CHECK'); END"))
    queued_run(monkeypatch)


def test_qa_worker_cannot_finalize_after_lease_takeover(monkeypatch):
    from datetime import datetime, timedelta, timezone
    run_id = queued_run(monkeypatch)
    worker = DurableJobWorker()
    old_job = worker._claim_one()
    assert old_job["lease_epoch"] == 1

    def old_execution_returns(*args, **kwargs):
        with db.engine.begin() as conn:
            conn.execute(update(job_table).where(job_table.c.id == old_job["id"]).values(lease_until=datetime.now(timezone.utc) - timedelta(seconds=1)))
        replacement = worker._claim_one()
        assert replacement["lease_epoch"] == 2
        with db.engine.begin() as conn:
            conn.execute(update(qa_run_table).where(qa_run_table.c.id == run_id).values(status="RUNNING"))
        # 原 epoch-1 执行晚到返回时，replacement 已经拥有 run；旧执行结果必须被 fencing 条件拒绝。
        QaService._finalize_run(ACTOR.tenant_id, run_id, "COMPLETED", "FINISHED", "COMPLETED", {"answer": "stale epoch-1 result"})

    monkeypatch.setattr(QaService, "_execute_run", old_execution_returns)
    worker._run_qa(old_job)
    with db.engine.connect() as conn:
        row = conn.execute(select(job_table).where(job_table.c.id == old_job["id"])).one()
        assert row.lease_epoch == 2
        assert row.status == "RUNNING", "Expired QA worker finalized replacement's job"


def test_purge_removes_document_from_agent_checkpoint(monkeypatch, tmp_path):
    doc = queue_document(monkeypatch, tmp_path)
    worker = DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    run_id = queued_run(monkeypatch)
    secret = "Milvus private company knowledge."
    with db.engine.begin() as conn:
        build_id = conn.execute(select(index_build_table.c.id)).scalar_one()
        conn.execute(run_source_table.insert().values(tenant_id=ACTOR.tenant_id, id=str(uuid.uuid4()), run_id=run_id, document_id=doc["documentId"], build_id=build_id))
    state = {"messages": [{"role": "tool", "content": secret}], "evidence": [{"documentId": doc["documentId"], "text": secret}]}
    if (ROOT / "backend/app/qa/job_checkpointer.py").exists():
        from app.qa.job_checkpointer import JobCheckpointSaver
        from langgraph.checkpoint.base import empty_checkpoint
        saver = JobCheckpointSaver()
        config = {"configurable": {"tenant_id": ACTOR.tenant_id, "thread_id": run_id}}
        cp = empty_checkpoint()
        cp["channel_values"] = state
        saver.put(config, cp, {"source": "update", "step": 0, "parents": {}}, {})
        def read_state():
            saved = saver.get_tuple(config)
            return saved.checkpoint["channel_values"] if saved else None
    else:
        from app.qa.checkpoint import save_agent_checkpoint, load_agent_checkpoint
        save_agent_checkpoint(ACTOR.tenant_id, run_id, state=state, next_node="model", deadline_epoch=9999999999)
        read_state = lambda: load_agent_checkpoint(ACTOR.tenant_id, run_id)
    lifecycle(doc["documentId"], "trash")
    lifecycle(doc["documentId"], "purge")
    with db.engine.connect() as conn:
        assert conn.execute(select(document_table.c.status)).scalar_one() == "PURGED"
        assert conn.execute(select(qa_run_table.c.terminal_reason)).scalar_one() == "SOURCE_PURGED"
    assert secret not in str(read_state()), "PURGED document plaintext remains readable from checkpoint"
