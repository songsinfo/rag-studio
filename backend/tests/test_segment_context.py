"""Persisted context must retain both its prefix and authorization dependencies."""
import copy

import pytest
from sqlalchemy import select, update

from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import Actor
from app.models import (
    context_segment_table, document_table, index_build_table, kb_grant_table,
    knowledge_base_table, qa_run_table, run_dependency_table, run_source_table,
)
from app.qa.business_context import state_from_messages
from app.qa.context import assert_run_sources_readable, conversation_memory
from app.qa.model_gateway import ModelGateway
from app.qa.segment_context import append_completed_turn
from app.qa.service import QaService


TENANT = "tenant-demo-1"
ACTOR = Actor(tenant_id=TENANT, user_id="user-admin-1", admin=True, security_epoch=1)


def insert_run(conn, run_id, turn, question="followup"):
    conn.execute(qa_run_table.insert().values(
        tenant_id=TENANT, id=run_id, owner_id=ACTOR.user_id,
        conversation_id="conversation", segment_id="segment", turn_no=turn,
        mode="AGENT", status="RUNNING", question=question,
        idempotency_key=run_id, request_hash=b"hash",
    ))


def seed_segment(conn, messages, state=None):
    insert_run(conn, "prior", 1)
    insert_run(conn, "current", 2)
    conn.execute(context_segment_table.insert().values(
        tenant_id=TENANT, id="segment", conversation_id="conversation",
        context_payload={"messages": messages, "businessState": state,
                         "sourceRuns": ["prior"]},
        context_version=1, context_through_turn=1, context_last_run_id="prior",
    ))


def test_loaded_sources_survive_connection_close_and_block_revoked_replay():
    with get_db_connection() as conn:
        seed_segment(conn, [{"role": "user", "content": "policy"},
                            {"role": "assistant", "content": "private policy"}])
        conn.execute(knowledge_base_table.insert().values(
            tenant_id=TENANT, id="kb", name="Private", status="LIVE"))
        conn.execute(kb_grant_table.insert().values(
            tenant_id=TENANT, id="grant", kb_id="kb", principal_type="USER",
            user_id=ACTOR.user_id))
        conn.execute(document_table.insert().values(
            tenant_id=TENANT, id="doc", kb_id="kb", acl_mode="INHERIT",
            status="LIVE", active_build_id="build"))
        conn.execute(index_build_table.insert().values(
            tenant_id=TENANT, id="build", document_id="doc", version_id="ver",
            config_id="config", build_no=1, status="READY", capabilities={}))
        conn.execute(run_source_table.insert().values(
            tenant_id=TENANT, id="source", run_id="prior", document_id="doc",
            build_id="build", evidence_id="E1"))
        conn.commit()

    for _ in range(2):
        memory = conversation_memory(ACTOR, "conversation", "current")
        assert memory[-1]["content"] == "private policy"
    with get_db_connection() as conn:
        sources = conn.execute(select(run_source_table).where(
            run_source_table.c.run_id == "current")).fetchall()
        assert len(sources) == 1
        assert (sources[0].document_id, sources[0].purpose, sources[0].evidence_id) == (
            "doc", "HISTORY", None)
        deps = conn.execute(select(run_dependency_table).where(
            run_dependency_table.c.run_id == "current")).fetchall()
        assert len(deps) == 1 and deps[0].prior_run_id == "prior"
        conn.execute(update(document_table).where(document_table.c.id == "doc")
                     .values(status="DELETED"))
        conn.commit()

    # Both model entry and checkpoint-resume guard must reject the already-loaded text.
    for check in (lambda: assert_run_sources_readable(ACTOR, "current"),
                  lambda: ModelGateway.assert_run_sources_current(TENANT, "current")):
        with pytest.raises(ApiException) as exc:
            check()
        assert exc.value.code == "SOURCE_CHANGED"
    assert QaService.get_run(ACTOR, "current")["visibility"] == "HIDDEN"


def test_snapshot_prefix_survives_two_appends_and_is_replaced_on_compaction():
    state = state_from_messages([{"role": "user", "content": "user U123 coupon C456"}])
    with get_db_connection() as conn:
        seed_segment(conn, [{"role": "user", "content": "user U123 coupon C456"},
                            {"role": "assistant", "content": "x" * 5100}], state)
        append_completed_turn(conn, TENANT, "current", {"answer": "short"})
        row = conn.execute(select(context_segment_table)).first()
        prefix = copy.deepcopy(row.context_payload["messages"])
        version = row.context_version
        assert prefix[0]["kind"] == "context_snapshot"
        assert "U123" in prefix[0]["content"]
        for turn in (3, 4):
            insert_run(conn, f"run-{turn}", turn)
            append_completed_turn(conn, TENANT, f"run-{turn}", {"answer": "short"})
            row = conn.execute(select(context_segment_table)).first()
            messages = row.context_payload["messages"]
            assert messages[:len(prefix)] == prefix
            assert row.context_version == version
            prefix = copy.deepcopy(messages)
        insert_run(conn, "run-5", 5, question="q" * 2000)
        append_completed_turn(conn, TENANT, "run-5", {"answer": "a" * 3500})
        row = conn.execute(select(context_segment_table)).first()
        messages = row.context_payload["messages"]
        assert row.context_version == version + 1
        assert sum(m.get("kind") == "context_snapshot" for m in messages) == 1
        assert [m["role"] for m in messages[1:]] == ["user", "assistant"]
