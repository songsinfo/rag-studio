import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, and_, update, delete

from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import Actor, require_admin
from app.models import (
    tenant_table,
    app_user_table,
    user_group_table,
    group_member_table,
    audit_event_table,
)

router = APIRouter(prefix="/admin", tags=["admin-identity"])


class GroupMembersRequest(BaseModel):
    userIds: List[str]
    expectedSecurityEpoch: int


@router.get("/users")
def list_users(actor: Actor = Depends(require_admin)):
    with get_db_connection() as conn:
        rows = conn.execute(
            select(
                app_user_table.c.id,
                app_user_table.c.login_name.label("loginName"),
                app_user_table.c.is_admin.label("admin"),
                app_user_table.c.status,
            ).where(app_user_table.c.tenant_id == actor.tenant_id)
        ).fetchall()
        return [
            {
                "id": str(r.id),
                "loginName": str(r.loginName),
                "admin": bool(r.admin),
                "status": str(r.status),
            }
            for r in rows
        ]


@router.get("/groups")
def list_groups(actor: Actor = Depends(require_admin)):
    with get_db_connection() as conn:
        tenant = conn.execute(
            select(tenant_table.c.security_epoch).where(tenant_table.c.id == actor.tenant_id)
        ).first()
        epoch = int(tenant.security_epoch) if tenant else 0

        groups = conn.execute(
            select(
                user_group_table.c.id,
                user_group_table.c.name,
            ).where(user_group_table.c.tenant_id == actor.tenant_id)
        ).fetchall()

        result = []
        for g in groups:
            members = conn.execute(
                select(group_member_table.c.user_id).where(
                    and_(
                        group_member_table.c.tenant_id == actor.tenant_id,
                        group_member_table.c.group_id == g.id,
                    )
                )
            ).fetchall()
            result.append({
                "id": str(g.id),
                "name": str(g.name),
                "securityEpoch": epoch,
                "userIds": [str(m.user_id) for m in members],
            })
        return result


@router.put("/groups/{group_id}/members")
def update_group_members(
    group_id: str,
    payload: GroupMembersRequest,
    actor: Actor = Depends(require_admin),
):
    if len(payload.userIds) > 1000:
        raise ApiException(400, "INVALID_INPUT", "userIds 数量超标")

    now = datetime.now(timezone.utc)
    with get_db_connection() as conn:
        grp = conn.execute(
            select(user_group_table.c.id).where(
                and_(
                    user_group_table.c.tenant_id == actor.tenant_id,
                    user_group_table.c.id == group_id,
                )
            )
        ).first()
        if not grp:
            raise ApiException(404, "RESOURCE_NOT_FOUND", "用户组不存在")

        # 使用 tenant.security_epoch 做乐观锁校验：权限更新必须基于最新安全版本，避免并发修改互相覆盖或旧权限被重新写回。
        res = conn.execute(
            update(tenant_table)
            .where(
                and_(
                    tenant_table.c.id == actor.tenant_id,
                    tenant_table.c.security_epoch == payload.expectedSecurityEpoch,
                )
            )
            .values(
                security_epoch=tenant_table.c.security_epoch + 1,
            )
        )
        if res.rowcount != 1:
            raise ApiException(409, "VERSION_CONFLICT", "数据已变化，请刷新后重试")

        unique_users = list(dict.fromkeys(payload.userIds))
        if unique_users:
            existing_users = conn.execute(
                select(app_user_table.c.id).where(
                    and_(
                        app_user_table.c.tenant_id == actor.tenant_id,
                        app_user_table.c.id.in_(unique_users),
                    )
                )
            ).fetchall()
            existing_set = {str(u.id) for u in existing_users}
            for uid in unique_users:
                if uid not in existing_set:
                    raise ApiException(400, "INVALID_INPUT", f"用户 {uid} 不存在")

        conn.execute(
            delete(group_member_table).where(
                and_(
                    group_member_table.c.tenant_id == actor.tenant_id,
                    group_member_table.c.group_id == group_id,
                )
            )
        )
        for uid in unique_users:
            conn.execute(
                group_member_table.insert().values(
                    tenant_id=actor.tenant_id,
                    id=str(uuid.uuid4()),
                    group_id=group_id,
                    user_id=uid,
                    created_at=now,
                    updated_at=now,
                )
            )

        conn.execute(
            audit_event_table.insert().values(
                tenant_id=actor.tenant_id,
                id=str(uuid.uuid4()),
                actor_id=actor.user_id,
                action="GROUP_MEMBERS_UPDATE",
                object_type="GROUP",
                object_id=group_id,
                outcome="SUCCESS",
                metadata_json={},
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

        return {"id": group_id, "updated": True}
