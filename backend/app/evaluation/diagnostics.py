"""Private, evaluation-only retrieval diagnostics.

Artifacts are not sent to Langfuse or regular logs; the event stores only the
opaque storage key. All retrieval IDs are reauthorized before capture and read.
"""
import json
import hashlib
import logging
import os
from typing import Any

from sqlalchemy import and_, select

from app.db import get_db_connection
from app.identity.deps import assert_hits_still_readable
from app.knowledge.artifacts import save_artifact, read_artifact
from app.models import qa_run_table, run_event_table, run_source_table, chunk_table, app_user_table

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 1


def candidate_snapshot(stage: str, candidates: list[dict], *, limit: int,
                       retrieval_index: int, context: bool = False,
                       route_index: int | None = None) -> dict:
    """Canonical stable order; missing offsets remain explicitly unknown."""
    selected = candidates[:max(0, limit)]
    items = []
    for rank, hit in enumerate(selected, 1):
        units = hit.get("sourceUnits") or []
        items.append({
            "rank": rank,
            "chunkId": hit.get("chunkId"),
            "documentId": hit.get("documentId"),
            "buildId": hit.get("buildId"),
            "indexEpoch": hit.get("indexEpoch"),
            "textSha256": hashlib.sha256(hit["text"].encode("utf-8")).hexdigest()
                if isinstance(hit.get("text"), str) else None,
            "parentChunkId": hit.get("parentChunkId"),
            "sourceType": hit.get("sourceType", "DOCUMENT"),
            "sourceUnits": [
                {"unitId": unit.get("unitId"), "contextStart": unit.get("contextStart"),
                 "contextEnd": unit.get("contextEnd"), "contextMapped": unit.get("contextMapped", False),
                 "location": unit.get("location")}
                for unit in units if isinstance(unit, dict)
            ],
        })
    return {"schemaVersion": _SCHEMA_VERSION, "stage": stage,
            "k": limit,
            "retrievalIndex": retrieval_index, "routeIndex": route_index,
            "candidateCount": len(candidates),
            "capturedCount": len(items), "complete": len(candidates) <= limit,
            "items": items, "contextState": "RETRIEVAL_OUTPUT" if context else "CANDIDATE_STAGE"}


def capture_eval_stage(ctx: Any, stage: str, candidates: list[dict],
                       limit: int, *, context: bool = False,
                       route_index: int | None = None):
    """Best-effort side path: never mutates search ordering, ACL, or QA status."""
    if not ctx.run_id:
        return
    try:
        with get_db_connection() as conn:
            run = conn.execute(select(qa_run_table.c.owner_id, qa_run_table.c.budget_json).where(and_(
                qa_run_table.c.tenant_id == ctx.actor.tenant_id,
                qa_run_table.c.id == ctx.run_id))).first()
            budget = (run.budget_json or {}) if run else {}
            scope = budget.get("evaluationScope") or {}
            if not scope or not run or str(run.owner_id) != ctx.actor.user_id:
                return
        from app.evaluation.catalog import EvaluationCatalog
        EvaluationCatalog.check_snapshot(ctx.actor, scope["snapshotId"], check_current=True)
        # Do not persist an unapproved candidate (including orphan or stale builds).
        with get_db_connection() as conn:
            assert_hits_still_readable(conn, ctx.actor, candidates)
        data = candidate_snapshot(stage, candidates, limit=limit,
                                  retrieval_index=ctx.retrieval_index, context=context,
                                  route_index=route_index)
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        key, _, _ = save_artifact(ctx.actor.tenant_id, "eval-retrieval-diagnostic.json", encoded)
        # No source ID or text appears in run_event; just an opaque private artifact locator.
        if ctx.record_event_fn:
            ctx.record_event_fn(ctx.actor.tenant_id, ctx.run_id, "EVAL_DIAG_ARTIFACT", {
                "schemaVersion": _SCHEMA_VERSION, "retrievalIndex": ctx.retrieval_index,
                "stage": stage, "routeIndex": route_index, "artifactKey": key,
                "complete": data["complete"], "candidateCount": data["candidateCount"],
                "capturedCount": data["capturedCount"]})
    except Exception:
        logger.warning("eval_diagnostic_capture_skipped")


def read_eval_diagnostics(actor, run_id: str) -> list[dict]:
    """Admin-only, persona-rechecked access to previously persisted artifacts."""
    if not actor.admin:
        from app.errors import ApiException
        raise ApiException(403, "OPERATION_FORBIDDEN", "仅管理员可读取评估诊断")
    from app.qa.service import QaService
    with get_db_connection() as conn:
        run = conn.execute(select(qa_run_table.c.owner_id, qa_run_table.c.budget_json).where(and_(
            qa_run_table.c.tenant_id == actor.tenant_id,
            qa_run_table.c.id == run_id))).first()
        if not run:
            from app.errors import ApiException
            raise ApiException(404, "RESOURCE_NOT_FOUND", "运行不存在")
        owner = conn.execute(select(app_user_table.c.status).where(and_(
            app_user_table.c.tenant_id == actor.tenant_id,
            app_user_table.c.id == run.owner_id))).first()
        if not owner or owner.status != "ACTIVE":
            from app.errors import ApiException
            raise ApiException(403, "OPERATION_FORBIDDEN", "评估 persona 已停用")
        budget = run.budget_json or {}
        scope = budget.get("evaluationScope")
        if not scope:
            from app.errors import ApiException
            raise ApiException(403, "OPERATION_FORBIDDEN", "不是评估运行")
        events = conn.execute(select(run_event_table.c.payload).where(and_(
            run_event_table.c.tenant_id == actor.tenant_id,
            run_event_table.c.run_id == run_id,
            run_event_table.c.stage == "EVAL_DIAG_ARTIFACT")).order_by(
                run_event_table.c.sequence_no.asc())).fetchall()
    from app.identity.deps import Actor
    from app.evaluation.catalog import EvaluationCatalog
    persona = Actor(actor.tenant_id, str(run.owner_id), False)
    EvaluationCatalog.check_snapshot(persona, scope["snapshotId"], check_current=True)
    detail = QaService.get_run(persona, run_id)
    if detail.get("visibility") == "HIDDEN":
        from app.errors import ApiException
        raise ApiException(403, "SOURCE_CHANGED", "权限或来源已改变")
    result = []
    for row in events:
        key = (row.payload or {}).get("artifactKey")
        if not key:
            continue
        try:
            data = json.loads(read_artifact(str(key)))
            items = data.get("items") or []
            with get_db_connection() as conn:
                assert_hits_still_readable(conn, persona, [
                    {"documentId": item.get("documentId"), "buildId": item.get("buildId"),
                     "chunkId": item.get("chunkId"), "indexEpoch": item.get("indexEpoch")}
                    for item in items])
            result.append(data)
        except Exception:
            result.append({"schemaVersion": _SCHEMA_VERSION, "status": "UNAVAILABLE"})
    return result


def model_context_snapshot(payload, sources, chunks):
    """Describe actual outgoing messages; never reconstruct from final citations.

    Exact ranges refer to a unique substring of the registered canonical chunk.
    Transformed/ambiguous text and plain history remain explicitly unknown.
    """
    messages = payload.get("messages") or []
    groups, items = [], []
    unmapped = 0

    def evidence_values(value):
        if isinstance(value, list):
            for child in value:
                yield from evidence_values(child)
        elif isinstance(value, dict):
            if value.get("evidenceId") and ("text" in value or "excerpt" in value):
                yield value
            else:
                for child in value.values():
                    yield from evidence_values(child)

    for index, message in enumerate(messages):
        content = message.get("content") or ""
        groups.append({"messageIndex": index, "role": message.get("role"),
                       "contentBytes": len(str(content).encode("utf-8")),
                       "historyMapping": "UNKNOWN"})
        try:
            structured = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError):
            continue
        for evidence in evidence_values(structured):
            matches = [s for s in sources if s["evidenceId"] == evidence.get("evidenceId")
                       and s["documentId"] == evidence.get("documentId")
                       and s["buildId"] == evidence.get("buildId")]
            if not matches:
                unmapped += 1
                continue
            source = matches[0]
            provenance = source.get("span") or {}
            excerpt = str(evidence.get("text") or evidence.get("excerpt") or "")
            sent_chars = len(excerpt)
            wrapper_chars = 0
            # The existing guardrail adds only these literal boundaries. Map the
            # retained document characters, including a prefix clipped by the
            # Agent's excerpt budget; arbitrary transformations remain unknown.
            prefix, suffix = '<retrieved_document trust="untrusted-data">\n', '\n</retrieved_document>'
            if "guardrail" in provenance and excerpt.startswith(prefix):
                wrapper_chars = len(prefix)
                excerpt = excerpt[len(prefix):]
                if excerpt.endswith(suffix):
                    excerpt = excerpt[:-len(suffix)]
            candidate_ids = [provenance.get("parentChunkId"), provenance.get("chunkId")]
            mapped = []
            for chunk_id in dict.fromkeys(candidate_ids):
                text = chunks.get((source["buildId"], chunk_id), "")
                if excerpt and text.count(excerpt) == 1:
                    start = text.index(excerpt)
                    mapped.append((chunk_id, start, start + len(excerpt)))
            chunk_id, start, end = mapped[0] if mapped else (None, None, None)
            items.append({"messageIndex": index, "evidenceId": source["evidenceId"],
                          "documentId": source["documentId"], "buildId": source["buildId"],
                          "chunkId": provenance.get("chunkId"),
                          "parentChunkId": provenance.get("parentChunkId"),
                          "contextChunkId": chunk_id, "contextStart": start, "contextEnd": end,
                          "mappingStatus": "EXACT" if mapped else "UNKNOWN",
                          "truncated": evidence.get("truncated") is True,
                          "sentChars": sent_chars, "wrapperPrefixChars": wrapper_chars, "sourceUnits": [
                              {"unitId": unit.get("unitId"), "location": unit.get("location"),
                               "contextStart": max(start, unit["contextStart"]),
                               "contextEnd": min(end, unit["contextEnd"])}
                              for unit in provenance.get("sourceUnits", [])
                              if mapped and chunk_id == (provenance.get("parentChunkId") or provenance.get("chunkId"))
                              and unit.get("contextMapped") is True
                              and type(unit.get("contextStart")) is int
                              and type(unit.get("contextEnd")) is int
                              and max(start, unit["contextStart"]) < min(end, unit["contextEnd"])
                          ]})
    return {"schemaVersion": 1, "stage": "MODEL_CONTEXT", "contextState": "ACTUAL_SEND",
            "offsetUnit": "UNICODE_CODEPOINT", "messageGroups": groups, "items": items,
            "messageCount": len(messages), "toolSchemaCount": len(payload.get("tools") or []),
            "historyMapping": "UNKNOWN", "unmappedEvidenceCount": unmapped,
            "complete": False, "sourceRangesComplete": unmapped == 0 and
                all(i["mappingStatus"] == "EXACT" for i in items)}


def capture_model_context(tenant_id, run_id, call_id, purpose, payload):
    """Return whether private I/O ran, so the caller rechecks before sending."""
    if not run_id:
        return False
    try:
        from app.identity.deps import Actor
        from app.evaluation.catalog import EvaluationCatalog
        from app.qa.service import QaService
        with get_db_connection() as conn:
            run = conn.execute(select(qa_run_table.c.owner_id, qa_run_table.c.budget_json).where(and_(
                qa_run_table.c.tenant_id == tenant_id, qa_run_table.c.id == run_id))).first()
            scope = (run.budget_json or {}).get("evaluationScope") if run else None
            if not scope:
                return False
            user = conn.execute(select(app_user_table.c.status).where(and_(
                app_user_table.c.tenant_id == tenant_id, app_user_table.c.id == run.owner_id))).first()
            if not user or user.status != "ACTIVE":
                return True
            actor = Actor(tenant_id, str(run.owner_id), False)
            rows = conn.execute(select(run_source_table).where(and_(
                run_source_table.c.tenant_id == tenant_id, run_source_table.c.run_id == run_id,
                run_source_table.c.source_type != "TOOL"))).fetchall()
            sources = [{"evidenceId": r.evidence_id, "documentId": str(r.document_id),
                        "buildId": str(r.build_id), "span": r.span_json or {}} for r in rows]
            assert_hits_still_readable(conn, actor, sources)
            ids = {cid for s in sources for cid in
                   (s["span"].get("chunkId"), s["span"].get("parentChunkId")) if cid}
            chunks = conn.execute(select(chunk_table.c.id, chunk_table.c.build_id, chunk_table.c.text_content)
                                  .where(and_(chunk_table.c.tenant_id == tenant_id,
                                              chunk_table.c.id.in_(ids)))).fetchall() if ids else []
        EvaluationCatalog.check_snapshot(actor, scope["snapshotId"], check_current=True)
        data = model_context_snapshot(payload, sources, {(str(c.build_id), str(c.id)): c.text_content for c in chunks})
        data.update({"callId": call_id, "purpose": purpose})
        key, _, _ = save_artifact(tenant_id, "eval-model-context.json",
                                 json.dumps(data, ensure_ascii=False).encode("utf-8"))
        QaService._record_event(tenant_id, run_id, "EVAL_DIAG_ARTIFACT", {
            "schemaVersion": 1, "stage": "MODEL_CONTEXT", "callId": call_id, "artifactKey": key,
        })
        return True
    except Exception:
        logger.warning("eval_model_context_capture_skipped")
        return True
