"""Explicit, best-effort synchronization of existing evaluation results to Langfuse.

Does not run a judge or change any authoritative DB rows. No text leaves the API.
"""
import hashlib
import logging
import os
import re

from sqlalchemy import and_, select

from app.db import get_db_connection
from app.errors import ApiException
from app.evaluation.catalog import EvaluationCatalog
from app.evaluation.runner import EvaluationRunner
from app.models import eval_run_table, eval_result_table, eval_review_table, run_event_table
from app.telemetry import configured_client

logger = logging.getLogger(__name__)
_TRACE_RE = re.compile(r"^[0-9a-f]{32}$")
_CATEGORICAL = {"correctness", "answerRelevance", "expectationAssessment"}
_NUMERIC = {"hit", "recall", "mrr", "ndcg", "factCoverage", "faithfulness"}
_SKIP_VERDICTS = {"NOT_EVALUATED", "NOT_APPLICABLE", "", "NONE"}


def score_id(target, result_id, dimension, metric_version, revision):
    data = "|".join(map(str, (target, result_id, dimension, metric_version, revision)))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]


def _create_scores(result, reviews, trace_id, target, rubric_version):
    judge = result.get("judge") or {}
    metrics = result.get("metrics") or {}
    numeric = {name: metrics.get(name) for name in _NUMERIC}
    for name in ("factCoverage", "faithfulness"):
        if numeric[name] is None:
            numeric[name] = judge.get(name)
    automatic = result.get("automaticJudgment") or {}
    for name in sorted(_CATEGORICAL):
        value = automatic.get(name)
        if isinstance(value, str) and value not in _SKIP_VERDICTS and re.fullmatch(r"[A-Z_]{2,40}", value):
            yield {"name": "automatic." + name + "." + rubric_version, "value": value,
                   "data_type": "CATEGORICAL", "revision": "automatic"}
    for name, value in sorted(numeric.items()):
        if type(value) in (int, float) and 0 <= float(value) <= 1 and float(value) == float(value):
            yield {"name": "automatic." + name + ".v1", "value": float(value),
                   "data_type": "NUMERIC", "revision": "automatic"}
    for review in reviews:
        for name, value in sorted((review.get("labels") or {}).items()):
            if name in _CATEGORICAL and isinstance(value, str) and value not in _SKIP_VERDICTS and re.fullmatch(r"[A-Z_]{2,40}", value):
                yield {"name": "human." + name + "." + rubric_version,
                       "value": value, "data_type": "CATEGORICAL",
                       "revision": str(review["id"])}


def _final_batch_trace_id(batch_events, target: str):
    """Only the latest recorded physical QA attempt can own the final answer.

    Do not fall back to an older sampled attempt when the winning lease was
    un-sampled, malformed, or mapped to another Langfuse project.
    """
    if not batch_events:
        return None
    final = batch_events[0].payload or {}
    trace_id = final.get("traceId")
    if (final.get("sampled") is True and final.get("targetInstance") == target
            and isinstance(trace_id, str) and _TRACE_RE.fullmatch(trace_id)):
        return trace_id
    return None


def sync_evaluation_scores(actor, eval_run_id: str):
    EvaluationCatalog.require_admin(actor)
    client = configured_client()
    if client is None:
        return {"sent": 0, "skipped": 0, "failed": 0, "status": "DISABLED"}
    with get_db_connection() as conn:
        ev = conn.execute(select(eval_run_table).where(and_(
            eval_run_table.c.tenant_id == actor.tenant_id,
            eval_run_table.c.id == eval_run_id))).first()
    if not ev:
        raise ApiException(404, "RESOURCE_NOT_FOUND", "评估运行不存在")
    EvaluationCatalog.check_snapshot(actor, str(ev.snapshot_id), check_current=True)
    rubric_version = str((ev.manifest_json or {}).get("rubricVersion") or "v1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", rubric_version):
        rubric_version = "v1"
    target = os.getenv("RAG_LANGFUSE_TARGET_ID", "private")
    results = EvaluationRunner.results(actor, eval_run_id)
    sent = skipped = failed = 0
    for item in results:
        if item.get("status") != "COMPLETED":
            skipped += 1
            continue
        qa_id = item.get("qaRunId")
        if not qa_id:
            skipped += 1
            continue
        # Revalidate the actual final QA run and all cited sources under its persona.
        # An administrator's own ACL must not substitute for the model-call persona.
        from app.qa.service import QaService
        try:
            from app.models import qa_run_table, app_user_table
            from app.identity.deps import Actor
            with get_db_connection() as conn:
                row = conn.execute(select(qa_run_table.c.owner_id).where(and_(
                    qa_run_table.c.tenant_id == actor.tenant_id, qa_run_table.c.id == qa_id))).first()
                user = conn.execute(select(app_user_table.c.status).where(and_(
                    app_user_table.c.tenant_id == actor.tenant_id,
                    app_user_table.c.id == str(row.owner_id) if row else ""))).first()
            if not row or not user or user.status != "ACTIVE":
                skipped += 1
                continue
            persona = Actor(actor.tenant_id, str(row.owner_id), False)
            detail = QaService.get_run(persona, qa_id)
            if (detail.get("visibility") == "HIDDEN" or detail.get("status") not in
                    {"COMPLETED", "PARTIAL", "CONFLICT", "NO_EVIDENCE", "SUCCEEDED"}):
                skipped += 1
                continue
            with get_db_connection() as conn:
                batch_events = conn.execute(select(run_event_table.c.payload).where(and_(
                    run_event_table.c.tenant_id == actor.tenant_id,
                    run_event_table.c.run_id == qa_id,
                    run_event_table.c.stage == "OBSERVABILITY_TRACE",
                )).order_by(run_event_table.c.sequence_no.desc())).fetchall()
            trace_id = _final_batch_trace_id(batch_events, target)
            if trace_id is None:
                skipped += 1
                continue
            for score in _create_scores(item, item.get("reviews") or [], trace_id, target, rubric_version):
                revision = score.pop("revision")
                name = score["name"]
                try:
                    client.create_score(trace_id=trace_id, score_id=score_id(target, item["id"], name,
                                       rubric_version, revision), **score)
                    sent += 1
                except Exception:
                    failed += 1
                    logger.warning("langfuse_score_enqueue_failed")
        except Exception:
            skipped += 1
            logger.warning("langfuse_score_permission_or_mapping_unavailable")
    # Enqueued != confirmed received. An admin may retry with the same IDs.
    return {"sent": sent, "skipped": skipped, "failed": failed, "status": "ENQUEUED_NOT_CONFIRMED"}
