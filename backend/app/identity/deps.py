"""身份与读取权限判定。

这里定义项目最核心的授权语义：
- Actor 永远带 tenant_id，所有查询必须先做租户隔离；
- “管理员”只代表维护权限，不自动等于可读取所有业务文档；
- 文档是否可读同时取决于用户状态、KB grant、文档 ACL、文档/KB 生命周期；
- 检索命中在真正使用前要再次验证 active build / index epoch，防止撤权或换代后的旧 hit 被继续使用。
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Request, Depends
from sqlalchemy import and_, exists, or_, select

from app.db import get_db_connection
from app.errors import ApiException
from app.identity.security import hash_token
from app.models import (
    app_user_table,
    auth_session_table,
    document_grant_table,
    document_table,
    group_member_table,
    index_build_table,
    kb_grant_table,
    knowledge_base_table,
    tenant_table,
)


@dataclass(frozen=True)
class Actor:
    """一次请求/任务执行时使用的安全主体快照。"""
    tenant_id: str
    user_id: str
    admin: bool
    security_epoch: int = 0


def get_current_actor(request: Request) -> Actor:
    """从 session cookie 解析当前有效用户，并读取最新 tenant security_epoch。"""
    session_token = request.cookies.get("RAG_SESSION")
    if not session_token:
        raise ApiException(401, "AUTH_REQUIRED", "请先登录")
    # 数据库只保存 token hash，不保存原始 session token。
    token_h = hash_token(session_token)
    with get_db_connection() as conn:
        stmt = (
            select(
                app_user_table.c.tenant_id,
                app_user_table.c.id.label("user_id"),
                app_user_table.c.is_admin,
                tenant_table.c.security_epoch,
            )
            .select_from(
                auth_session_table.join(
                    app_user_table,
                    and_(
                        app_user_table.c.tenant_id == auth_session_table.c.tenant_id,
                        app_user_table.c.id == auth_session_table.c.user_id,
                    ),
                ).join(tenant_table, tenant_table.c.id == app_user_table.c.tenant_id)
            )
            .where(
                and_(
                    auth_session_table.c.token_hash == token_h,
                    auth_session_table.c.revoked_at.is_(None),
                    auth_session_table.c.expires_at > datetime.now(timezone.utc),
                    app_user_table.c.status == "ACTIVE",
                )
            )
        )
        row = conn.execute(stmt).first()
        if not row:
            raise ApiException(401, "AUTH_REQUIRED", "登录状态已失效")
        # 每次请求都从数据库重建 Actor，用户禁用/会话吊销可以立即生效。
        return Actor(
            tenant_id=str(row.tenant_id),
            user_id=str(row.user_id),
            admin=bool(row.is_admin),
            security_epoch=int(row.security_epoch or 0),
        )


def require_admin(actor: Actor = Depends(get_current_actor)) -> Actor:
    """仅校验维护权限；不要把它误用成文档读取权限。"""
    if not actor.admin:
        raise ApiException(403, "OPERATION_FORBIDDEN", "需要维护权限")
    return actor


def principal_matches(grant_table, actor: Actor):
    """生成 grant 主体匹配条件：ALL / 指定用户 / 用户所在组。"""
    return or_(
        grant_table.c.principal_type == "ALL",
        grant_table.c.user_id == actor.user_id,
        exists().where(
            and_(
                group_member_table.c.tenant_id == actor.tenant_id,
                group_member_table.c.user_id == actor.user_id,
                group_member_table.c.group_id == grant_table.c.group_id,
            )
        ),
    )


def _access_predicates(actor: Actor):
    """构造统一的文档读取谓词，避免各业务模块自己拼 ACL 产生语义漂移。"""
    user_active = exists().where(
        and_(
            app_user_table.c.tenant_id == actor.tenant_id,
            app_user_table.c.id == actor.user_id,
            app_user_table.c.status == "ACTIVE",
        )
    )
    # KB 必须 LIVE 且当前主体命中 kb_grant。
    kb_accessible = exists().where(
        and_(
            knowledge_base_table.c.tenant_id == actor.tenant_id,
            knowledge_base_table.c.id == document_table.c.kb_id,
            knowledge_base_table.c.status == "LIVE",
            exists().where(
                and_(
                    kb_grant_table.c.tenant_id == actor.tenant_id,
                    kb_grant_table.c.kb_id == knowledge_base_table.c.id,
                    principal_matches(kb_grant_table, actor),
                )
            ),
        )
    )
    # INHERIT 继承 KB 权限；非继承模式要求命中文档自己的 grant。
    doc_acl_ok = or_(
        document_table.c.acl_mode == "INHERIT",
        exists().where(
            and_(
                document_grant_table.c.tenant_id == actor.tenant_id,
                document_grant_table.c.document_id == document_table.c.id,
                principal_matches(document_grant_table, actor),
            )
        ),
    )
    return user_active, kb_accessible, doc_acl_ok


def can_read_document(conn, actor: Actor, doc_id: str) -> bool:
    """判断当前 Actor 此刻是否仍能读取一篇 LIVE 文档。"""
    user_active, kb_accessible, doc_acl_ok = _access_predicates(actor)
    stmt = select(document_table.c.id).where(
        and_(
            document_table.c.tenant_id == actor.tenant_id,
            document_table.c.id == doc_id,
            document_table.c.status == "LIVE",
            user_active,
            kb_accessible,
            doc_acl_ok,
        )
    )
    return conn.execute(stmt).first() is not None


def get_readable_build_records(
    conn,
    actor: Actor,
    kb_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """返回 Actor 当前可读且 READY 的 active build 及其能力信息。"""
    user_active, kb_accessible, doc_acl_ok = _access_predicates(actor)
    stmt = (
        select(
            document_table.c.active_build_id,
            document_table.c.id.label("document_id"),
            document_table.c.kb_id,
            index_build_table.c.version_id,
            index_build_table.c.fence,
            index_build_table.c.model_fingerprint,
            index_build_table.c.embedding_dims,
            index_build_table.c.capabilities,
        )
        .select_from(
            document_table.join(
                index_build_table,
                and_(
                    index_build_table.c.tenant_id == document_table.c.tenant_id,
                    index_build_table.c.id == document_table.c.active_build_id,
                ),
            )
        )
        .where(
            and_(
                document_table.c.tenant_id == actor.tenant_id,
                document_table.c.status == "LIVE",
                # 只允许当前已发布的 active build；历史/未完成 build 不进入在线检索。
                document_table.c.active_build_id.isnot(None),
                index_build_table.c.status == "READY",
                user_active,
                kb_accessible,
                doc_acl_ok,
            )
        )
    )
    if kb_ids:
        stmt = stmt.where(document_table.c.kb_id.in_(kb_ids))
    if document_ids is not None:
        stmt = stmt.where(document_table.c.id.in_(document_ids))

    rows = conn.execute(stmt).fetchall()
    return [
        {
            "buildId": str(r.active_build_id),
            "documentId": str(r.document_id),
            "kbId": str(r.kb_id),
            "versionId": str(r.version_id),
            "indexEpoch": int(r.fence or 0),
            "modelFingerprint": str(r.model_fingerprint) if r.model_fingerprint else None,
            "embeddingDims": int(r.embedding_dims) if r.embedding_dims is not None else None,
            "capabilities": r.capabilities if isinstance(r.capabilities, dict) else {},
        }
        for r in rows
    ]


def get_readable_builds(conn, actor: Actor, kb_ids=None, document_ids=None) -> list:
    return [r["buildId"] for r in get_readable_build_records(conn, actor, kb_ids, document_ids)]


def assert_hits_still_readable(
    conn,
    actor: Actor,
    hits: List[Dict[str, Any]],
    kb_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
) -> None:
    """在候选真正被消费前重新校验权限、active build 和 index epoch。

    这是防 TOCTOU 的二次校验：检索发生后，ACL 或索引可能已经变化。
    """
    if not hits:
        return
    # 只查询候选涉及的文档，避免为了二次鉴权扫描整个租户。
    candidate_ids = list({str(h.get("documentId") or "") for h in hits})
    if document_ids is not None:
        candidate_ids = list(set(candidate_ids).intersection(document_ids))
    readable = get_readable_build_records(conn, actor, kb_ids, candidate_ids)
    # key 同时包含 buildId + documentId，防止只凭 documentId 接受过期 build。
    allowed = {
        (r["buildId"], r["documentId"]): int(r.get("indexEpoch") or 0)
        for r in readable
    }
    for hit in hits:
        build_id = str(hit.get("buildId") or "")
        document_id = str(hit.get("documentId") or "")
        expected_epoch = allowed.get((build_id, document_id))
        hit_epoch = hit.get("indexEpoch")
        if not build_id or not document_id or expected_epoch is None:
            raise ApiException(409, "SOURCE_CHANGED", "检索来源权限或生效版本已变化，请重新提问")
        # epoch/fence 用来识别同一 build ID 下的索引代次变化；不一致时要求重新检索。
        if expected_epoch > 0 and hit_epoch is not None and int(hit_epoch) != expected_epoch:
            raise ApiException(409, "SOURCE_CHANGED", "检索索引代次已变化，请重新提问")
