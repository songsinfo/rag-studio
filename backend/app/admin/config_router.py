import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, status
from sqlalchemy import select, and_, func, update

from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import Actor, get_current_actor
from app.models import config_revision_table, tenant_table, audit_event_table
from app.retrieval.config import validate_retrieval_payload

router = APIRouter(prefix="/admin/configs", tags=["admin-configs"])

VALID_KINDS = {"BUILD", "RETRIEVAL", "MODEL", "PROMPT", "LIMITS", "EVAL"}


@router.get("")
@router.get("/")
def list_configs(actor: Actor = Depends(get_current_actor)):
    if not actor.admin:
        raise ApiException(403, "ACCESS_DENIED", "需要管理员权限")
    with get_db_connection() as conn:
        stmt = (
            select(
                config_revision_table.c.id,
                config_revision_table.c.kind,
                config_revision_table.c.name,
                config_revision_table.c.revision,
                config_revision_table.c.status,
                config_revision_table.c.payload,
                config_revision_table.c.created_at.label("createdAt"),
            )
            .where(config_revision_table.c.tenant_id == actor.tenant_id)
            .order_by(config_revision_table.c.created_at.desc())
            .limit(100)
        )
        rows = conn.execute(stmt).fetchall()
        return [
            {
                "id": str(r.id),
                "kind": str(r.kind),
                "name": str(r.name),
                "revision": int(r.revision),
                "status": str(r.status),
                "payload": r.payload if isinstance(r.payload, dict) else json.loads(r.payload or "{}"),
                "createdAt": r.createdAt.isoformat() if r.createdAt else None,
            }
            for r in rows
        ]


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_config(body: Dict[str, Any], actor: Actor = Depends(get_current_actor)):
    if not actor.admin:
        raise ApiException(403, "ACCESS_DENIED", "需要管理员权限")

    kind = str(body.get("kind", "")).strip().upper()
    name = str(body.get("name", "")).strip()
    payload = body.get("payload")

    if kind not in VALID_KINDS or not name or len(name) > 128 or not isinstance(payload, dict):
        raise ApiException(400, "INVALID_CONFIG", "配置字段无效")
    if kind == "RETRIEVAL":
        validate_retrieval_payload(payload)
    if kind == "MODEL":
        alias = str(payload.get("modelAlias") or "").strip()
        currency = str(payload.get("currency", "CNY")).upper()
        try:
            input_price = float(payload.get("inputPricePerMillion"))
            output_price = float(payload.get("outputPricePerMillion"))
        except (TypeError, ValueError) as exc:
            raise ApiException(400, "INVALID_CONFIG", "MODEL 配置必须包含合法价格") from exc
        if (
            not alias
            or currency != "CNY"
            or input_price < 0
            or output_price < 0
        ):
            raise ApiException(400, "INVALID_CONFIG", "MODEL 配置必须包含 modelAlias、CNY 价格且价格非负")

    payload_json = json.dumps(payload, ensure_ascii=False)
    if len(payload_json) > 64000 or "api_key" in payload_json.lower() or "apikey" in payload_json.lower():
        raise ApiException(400, "INVALID_CONFIG", "配置不可包含密钥")

    sha256 = hashlib.sha256(payload_json.encode("utf-8")).digest()
    now = datetime.now(timezone.utc)
    config_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        rev_stmt = select(func.coalesce(func.max(config_revision_table.c.revision), 0) + 1).where(
            and_(
                config_revision_table.c.tenant_id == actor.tenant_id,
                config_revision_table.c.kind == kind,
                config_revision_table.c.name == name,
            )
        )
        revision = conn.execute(rev_stmt).scalar() or 1

        conn.execute(
            config_revision_table.insert().values(
                tenant_id=actor.tenant_id,
                id=config_id,
                kind=kind,
                name=name,
                revision=revision,
                sha256=sha256,
                payload=payload,
                status="DRAFT",
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    return {"id": config_id, "revision": revision, "status": "DRAFT"}


@router.post("/{config_id}/publish")
def publish_config(config_id: str, actor: Actor = Depends(get_current_actor)):
    if not actor.admin:
        raise ApiException(403, "ACCESS_DENIED", "需要管理员权限")

    now = datetime.now(timezone.utc)
    with get_db_connection() as conn:
        cfg = conn.execute(
            select(config_revision_table).where(
                and_(
                    config_revision_table.c.tenant_id == actor.tenant_id,
                    config_revision_table.c.id == config_id,
                )
            )
        ).first()
        if not cfg:
            raise ApiException(404, "RESOURCE_NOT_FOUND", "配置不存在")

        conn.execute(
            update(config_revision_table)
            .where(
                and_(
                    config_revision_table.c.tenant_id == actor.tenant_id,
                    config_revision_table.c.id == config_id,
                )
            )
            .values(status="PUBLISHED", updated_at=now)
        )

        if cfg.kind == "RETRIEVAL":
            conn.execute(
                update(tenant_table)
                .where(tenant_table.c.id == actor.tenant_id)
                .values(active_retrieval_config_id=config_id)
            )

        conn.execute(
            audit_event_table.insert().values(
                tenant_id=actor.tenant_id,
                id=str(uuid.uuid4()),
                actor_id=actor.user_id,
                action="CONFIG_PUBLISH",
                object_type="CONFIG",
                object_id=config_id,
                outcome="SUCCESS",
                metadata_json={"kind": cfg.kind, "name": cfg.name, "revision": cfg.revision},
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    return {"id": config_id, "status": "PUBLISHED"}
