"""问答历史与评估快照的授权边界。

这里的核心职责不是“拼历史文本”，而是保证历史内容在当前时刻仍然可读。
权限可能在一次会话开始后发生变化，所以历史回答再次进入模型上下文前必须重新校验，
不能因为它过去曾经可读就永久信任。
"""

from sqlalchemy import and_, select

from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import assert_hits_still_readable, can_read_document
from app.models import document_table, qa_run_table, run_source_table


def run_document_scope(conn, actor, run_id):
    """读取评估运行锁定的文档/build 范围，并验证快照仍未漂移。"""
    budget = conn.execute(select(qa_run_table.c.budget_json).where(and_(
        qa_run_table.c.tenant_id == actor.tenant_id, qa_run_table.c.id == run_id,
    ))).scalar() or {}
    # 普通在线问答没有 evaluationScope；只有固定评估快照才要求文档/build 完全一致。
    scope = budget.get("evaluationScope")
    if scope is None:
        return None
    documents = scope.get("documents", [])
    # 评估时锁定的不只是 documentId，还锁定 active build。
    # 同一文档重新入库后如果 build 变化，本次评估应失败，而不是偷偷测到新版本。
    expected = {str(d["documentId"]): str(d["buildId"]) for d in documents}
    rows = conn.execute(select(document_table.c.id, document_table.c.active_build_id).where(and_(
        document_table.c.tenant_id == actor.tenant_id,
        document_table.c.id.in_(expected), document_table.c.status == "LIVE",
    ))).fetchall()
    # 任一文档被删除、下线或 active_build_id 变化，都说明快照已经失效。
    if not expected or {str(r.id): str(r.active_build_id) for r in rows} != expected:
        raise ApiException(409, "SOURCE_CHANGED", "评测快照的文档或生效构建已变化")
    return list(expected)


def assert_run_sources_readable(actor, run_id):
    """在“再次读取/发送来源”前做当前权限校验。

    run_source 保存的是运行当时使用过的来源，但它不是永久授权凭证。
    当前 ACL 已撤销时必须 fail-closed。
    """
    with get_db_connection() as conn:
        document_ids = run_document_scope(conn, actor, run_id)
        rows = conn.execute(select(run_source_table.c.document_id, run_source_table.c.build_id,
                   run_source_table.c.source_type, run_source_table.c.purpose).where(and_(
            run_source_table.c.tenant_id == actor.tenant_id, run_source_table.c.run_id == run_id,
        )).distinct()).fetchall()
        # HISTORY 来源需要额外检查当前文档 ACL。
        # 这是为了防止历史回答在撤权后又被重新送入后续模型/Judge。
        for row in rows:
            if row.source_type != "TOOL" and row.purpose == "HISTORY" and not can_read_document(conn, actor, str(row.document_id)):
                raise ApiException(409, "SOURCE_CHANGED", "历史来源权限已变化")
        # 对实际命中的 document/build 再做一次“当前仍可读 + build 未漂移”的检查。
        assert_hits_still_readable(conn, actor, [
            {"documentId": r.document_id, "buildId": r.build_id} for r in rows
            if r.source_type != "TOOL" and (r.purpose != "HISTORY" or document_ids is not None)
        ], document_ids=document_ids)


def conversation_memory(actor, conversation_id, current_run_id, max_turns=4):
    """优先读取同 Segment 的授权持久化快照；无快照时回退 QaService._rag_history 旧历史裁剪。"""
    from app.qa.service import QaService
    with get_db_connection() as conn:
        segment_id = conn.execute(select(qa_run_table.c.segment_id).where(and_(
            qa_run_table.c.tenant_id == actor.tenant_id,
            qa_run_table.c.id == current_run_id,
            qa_run_table.c.conversation_id == conversation_id,
            qa_run_table.c.owner_id == actor.user_id,
        ))).scalar()
    # 找不到当前 run 所属 segment 时不跨段猜测历史，直接视为无历史。
    if not segment_id:
        return []
    from app.qa.segment_context import load_segment_context
    with get_db_connection() as conn:
        messages, meta = load_segment_context(conn, actor, str(segment_id), current_run_id)
        conn.commit()
    if meta["source"] == "PERSISTED":
        return messages
    return QaService._rag_history(actor, str(segment_id), current_run_id)
