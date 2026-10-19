"""Regression for stale QA worker entry, using REVIEW_ROOT checkout."""
import importlib.util
import os
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

spec = importlib.util.spec_from_file_location("stack_review", Path(__file__).with_name("test_stack_review.py"))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
setup_test_db = review.setup_test_db
tmp_path = review.tmp_path

from sqlalchemy import select, update, func
from app.models import job_table, qa_run_table, run_source_table, run_event_table, index_build_table


def test_delayed_old_worker_must_not_reset_replacement(monkeypatch, tmp_path):
    doc = review.queue_document(monkeypatch, tmp_path)
    worker = review.DurableJobWorker()
    worker._run_ingest(worker._claim_one())
    run_id = review.queued_run(monkeypatch)
    old_job = worker._claim_one()
    with review.db.engine.begin() as conn:
        conn.execute(update(job_table).where(job_table.c.id == old_job["id"]).values(lease_until=datetime.now(timezone.utc) - timedelta(seconds=1)))
    replacement = worker._claim_one()
    assert replacement["lease_epoch"] == old_job["lease_epoch"] + 1
    # replacement worker 已经启动并持久化了 source/event 状态，
    # 原进程则在进入 execute_queued_run 前被挂起；该场景用于验证晚到旧 worker 不会覆盖新 owner 状态。
    with review.db.engine.begin() as conn:
        build_id = conn.execute(select(index_build_table.c.id)).scalar_one()
        conn.execute(update(qa_run_table).where(qa_run_table.c.id == run_id).values(status="RUNNING", mode="RAG"))
        conn.execute(run_source_table.insert().values(tenant_id=review.ACTOR.tenant_id, id=str(uuid.uuid4()), run_id=run_id, document_id=doc["documentId"], build_id=build_id))
        conn.execute(run_event_table.insert().values(tenant_id=review.ACTOR.tenant_id, id=str(uuid.uuid4()), run_id=run_id, sequence_no=1, stage="RETRIEVED", payload={"replacement": True}))
    worker._run_qa(old_job)
    with review.db.engine.connect() as conn:
        status = conn.execute(select(qa_run_table.c.status).where(qa_run_table.c.id == run_id)).scalar_one()
        sources = conn.execute(select(func.count()).select_from(run_source_table).where(run_source_table.c.run_id == run_id)).scalar_one()
        events = conn.execute(select(func.count()).select_from(run_event_table).where(run_event_table.c.run_id == run_id)).scalar_one()
        actual = (status, sources, events)
    assert actual == ("RUNNING", 1, 1), f"Stale worker erased replacement state: {actual}"
