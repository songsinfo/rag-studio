"""Versioned conversation segment context. Only authorized prior run text is reused.

The segment is the authority and compaction boundary. An Agent run's Job
checkpoint remains separate for node-level replay. Never copy old E1 IDs into
the new run's evidence map or promote historical snapshots to system text.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import uuid

from sqlalchemy import and_, select, update

from app.errors import ApiException
from app.models import context_segment_table, qa_run_table, run_dependency_table, run_source_table
from app.identity.deps import assert_hits_still_readable, can_read_document
from app.qa.business_context import (
    normalize_business_state, state_from_messages, state_message, update_from_tool,
)
from app.qa.execution import qa_lease_is_current

MAX_CHARS = 5000


def load_segment_context(conn, actor, segment_id: str, current_run_id: str) -> Tuple[List[Dict], Dict]:
    row = conn.execute(select(context_segment_table).where(and_(
        context_segment_table.c.tenant_id == actor.tenant_id,
        context_segment_table.c.id == segment_id,
    ))).first()
    if row is None:
        raise ApiException(404, "RESOURCE_NOT_FOUND", "会话分段不存在")
    snapshot = row.context_payload if isinstance(row.context_payload, dict) else {}
    if not snapshot or int(row.context_through_turn or 0) == 0:
        return [], {"version": 0, "businessState": None, "source": "LEGACY"}
    current = conn.execute(select(qa_run_table).where(and_(
        qa_run_table.c.tenant_id == actor.tenant_id,
        qa_run_table.c.id == current_run_id,
        qa_run_table.c.owner_id == actor.user_id,
        qa_run_table.c.segment_id == segment_id,
    ))).first()
    if current is None or int(row.context_through_turn) >= int(current.turn_no):
        raise ApiException(409, "SOURCE_CHANGED", "上下文不属于当前问答的历史")
    if not qa_lease_is_current(conn, actor.tenant_id, current_run_id):
        raise ApiException(409, "JOB_LEASE_LOST", "问答任务租约已失效")
    source_runs = snapshot.get("sourceRuns") or []
    if len(source_runs) > 5000:
        raise ApiException(409, "SOURCE_CHANGED", "历史来源数量超出安全限制")
    inherited_documents = set()
    for rid in source_runs:
        if not isinstance(rid, str):
            raise ApiException(409, "SOURCE_CHANGED", "上下文来源格式无效")
        prior = conn.execute(select(qa_run_table.c.owner_id, qa_run_table.c.segment_id,
                                    qa_run_table.c.turn_no).where(and_(
            qa_run_table.c.tenant_id == actor.tenant_id, qa_run_table.c.id == rid,
        ))).first()
        if (prior is None or prior.owner_id != actor.user_id
                or str(prior.segment_id) != segment_id or prior.turn_no >= current.turn_no):
            raise ApiException(409, "SOURCE_CHANGED", "历史上下文来源已变化")
        documents = conn.execute(select(
            run_source_table.c.document_id, run_source_table.c.build_id,
        ).where(and_(run_source_table.c.tenant_id == actor.tenant_id,
                     run_source_table.c.run_id == rid,
                     run_source_table.c.document_id.is_not(None)))).fetchall()
        for source in documents:
            if not can_read_document(conn, actor, str(source.document_id)):
                raise ApiException(409, "SOURCE_CHANGED", "历史上下文授权已失效")
        if documents:
            assert_hits_still_readable(conn, actor, [
                {"documentId": str(s.document_id), "buildId": str(s.build_id)}
                for s in documents
            ])
        inherited_documents.update((s.document_id, s.build_id) for s in documents)
    messages = snapshot.get("messages") or []
    if not isinstance(messages, list) or any(not isinstance(m, dict) or
        m.get("role") not in ("user", "assistant") or
        not isinstance(m.get("content"), str) for m in messages):
        raise ApiException(409, "SOURCE_CHANGED", "上下文快照格式无效")
    # Persist dependencies before these messages enter a checkpoint or model request.
    # Callers commit this transaction; subsequent guards can then recheck revoked history.
    existing_runs = set(conn.execute(select(run_dependency_table.c.prior_run_id).where(and_(
        run_dependency_table.c.tenant_id == actor.tenant_id,
        run_dependency_table.c.run_id == current_run_id,
    ))).scalars())
    now = datetime.now(timezone.utc)
    for rid in set(source_runs) - existing_runs:
        conn.execute(run_dependency_table.insert().values(
            tenant_id=actor.tenant_id, id=str(uuid.uuid4()), run_id=current_run_id,
            prior_run_id=rid, created_at=now, updated_at=now,
        ))
    existing_documents = set(conn.execute(select(
        run_source_table.c.document_id, run_source_table.c.build_id,
    ).where(and_(
        run_source_table.c.tenant_id == actor.tenant_id,
        run_source_table.c.run_id == current_run_id,
        run_source_table.c.purpose == "HISTORY",
    ))).tuples())
    for document_id, build_id in inherited_documents - existing_documents:
        conn.execute(run_source_table.insert().values(
            tenant_id=actor.tenant_id, id=str(uuid.uuid4()), run_id=current_run_id,
            document_id=document_id, build_id=build_id, purpose="HISTORY",
            source_type="DOCUMENT", evidence_id=None, send_status="PREPARED",
            created_at=now, updated_at=now,
        ))
    return [dict(m) for m in messages], {
        "version": int(row.context_version or 0),
        "throughTurn": int(row.context_through_turn or 0),
        "boundaryTurn": int(row.context_boundary_turn or 0),
        "businessState": normalize_business_state(snapshot.get("businessState")),
        "source": "PERSISTED",
    }


def append_completed_turn(conn, tenant_id: str, run_id: str, result: Dict[str, Any]) -> None:
    """Called within the fenced success-finalization transaction; no partial writes."""
    run = conn.execute(select(qa_run_table).where(and_(
        qa_run_table.c.tenant_id == tenant_id, qa_run_table.c.id == run_id,
    ))).first()
    if not run or run.mode != "AGENT":
        return
    segment = conn.execute(select(context_segment_table).where(and_(
        context_segment_table.c.tenant_id == tenant_id,
        context_segment_table.c.id == run.segment_id,
    )).with_for_update()).first()
    if segment is None:
        raise ApiException(409, "SOURCE_CHANGED", "上下文分段不存在")
    if str(segment.context_last_run_id or "") == run_id:
        return
    if int(segment.context_through_turn or 0) >= int(run.turn_no):
        raise ApiException(409, "VERSION_CONFLICT", "上下文版本已被后续问答推进")
    previous = segment.context_payload if isinstance(segment.context_payload, dict) else {}
    messages = [dict(m) for m in previous.get("messages", [])]
    state = normalize_business_state(previous.get("businessState"))
    answer = str(result.get("answer") or "")[:4000]
    user = str(run.question or "")[:2000]
    # Historical text is data, not an instruction, and old evidence IDs are not
    # registered in a new run. Structured objects are scoped to this segment.
    new_pair = [{"role": "user", "content": user},
                {"role": "assistant", "content": answer}]
    state = state_from_messages([new_pair[0]], state, source_run_id=run_id)
    for ev in result.get("evidence", []):
        if ev.get("sourceType") == "TOOL" and ev.get("mock") is True:
            state = update_from_tool(state, ev.get("toolName", ""), {
                "status": "OK", "mock": True,
                "queryObjects": ev.get("queryObjects") or {},
                "queriedAt": ev.get("queriedAt"),
                "fixtureVersion": ev.get("fixtureVersion"),
            }, source_run_id=run_id, evidence_id=ev.get("evidenceId") or "")
    state["pending"] = [g for g in result.get("gaps", []) if isinstance(g, str)][:8]
    messages.extend(new_pair)
    source_runs = [rid for rid in previous.get("sourceRuns", []) if isinstance(rid, str)]
    source_runs.append(run_id)
    boundary = int(segment.context_boundary_turn or 0)
    version = int(segment.context_version or 0) or 1
    compressed = False
    if sum(len(str(m.get("content") or "")) for m in messages) > MAX_CHARS:
        # Keep recent full turns; preserve older explicit state as untrusted text.
        # Only replace snapshots at a compaction boundary, never on ordinary append.
        previous_snapshot = any(m.get("kind") == "context_snapshot" for m in messages)
        messages = [m for m in messages if m.get("kind") != "context_snapshot"]
        compressed = previous_snapshot
        while len(messages) > 2 and sum(len(str(m["content"])) for m in messages) > 3000:
            messages = messages[2:]
            compressed = True
        if compressed:
            version += 1
            boundary = int(run.turn_no) - len(messages) // 2
            messages.insert(0, state_message(state))
    # Preserve authorization dependencies even for compressed text. Any revoked
    # old source fails closed on replay; no privilege inherited from a snapshot.
    next_payload = {
        "schemaVersion": 1, "messages": messages,
        "businessState": state, "sourceRuns": list(dict.fromkeys(source_runs)),
        "lastRunId": run_id, "throughTurn": int(run.turn_no),
        "boundaryTurn": boundary,
    }
    result_update = conn.execute(update(context_segment_table).where(and_(
        context_segment_table.c.tenant_id == tenant_id,
        context_segment_table.c.id == run.segment_id,
        context_segment_table.c.context_through_turn < int(run.turn_no),
    )).values(
        context_payload=next_payload,
        context_version=version,
        context_through_turn=int(run.turn_no),
        context_boundary_turn=boundary,
        context_last_run_id=run_id,
        updated_at=datetime.now(timezone.utc),
    ))
    if result_update.rowcount != 1:
        raise ApiException(409, "VERSION_CONFLICT", "无法原子推进上下文版本")
