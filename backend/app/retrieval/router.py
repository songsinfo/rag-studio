import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_, update

from app.db import get_db_connection
from app.identity.deps import Actor, get_current_actor
from app.models import tenant_table, job_table
from app.retrieval.pipeline import RetrievalPipeline

router = APIRouter(tags=["retrieval"])


@router.get("/search")
def search(
    query: str = Query(..., min_length=1),
    kbIds: Optional[str] = Query(None),
    actor: Actor = Depends(get_current_actor),
):
    kb_list = [k.strip() for k in kbIds.split(",") if k.strip()] if kbIds else None
    with get_db_connection() as conn:
        config_id = conn.execute(
            select(tenant_table.c.active_retrieval_config_id).where(
                tenant_table.c.id == actor.tenant_id
            )
        ).scalar()
    now = datetime.now(timezone.utc)
    job_id = str(uuid.uuid4())
    request_hash = hashlib.sha256(
        f"{actor.user_id}:{query}:{kb_list}:{config_id}".encode("utf-8")
    ).digest()
    with get_db_connection() as conn:
        conn.execute(
            job_table.insert().values(
                tenant_id=actor.tenant_id,
                id=job_id,
                kind="SEARCH",
                target_id=actor.user_id,
                requested_by=actor.user_id,
                idempotency_key=job_id,
                request_hash=request_hash,
                status="RUNNING",
                stage="SEARCH",
                attempt=0,
                available_at=now,
                lease_epoch=0,
                checkpoint={"query": query},
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    try:
        result = RetrievalPipeline.retrieve(
            actor=actor,
            query=query,
            kb_ids=kb_list,
            config_id=str(config_id) if config_id else "cfg-enterprise-retrieval-v2",
            job_id=job_id,
            limit_override=10,
        )
    except Exception as exc:
        with get_db_connection() as conn:
            conn.execute(
                update(job_table)
                .where(
                    and_(
                        job_table.c.tenant_id == actor.tenant_id,
                        job_table.c.id == job_id,
                    )
                )
                .values(
                    status="FAILED",
                    stage="FINISHED",
                    error_code=getattr(exc, "code", type(exc).__name__),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            conn.commit()
        raise
    else:
        with get_db_connection() as conn:
            conn.execute(
                update(job_table)
                .where(
                    and_(
                        job_table.c.tenant_id == actor.tenant_id,
                        job_table.c.id == job_id,
                    )
                )
                .values(
                    status="COMPLETED",
                    stage="FINISHED",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            conn.commit()
    evidence = []
    for i, hit in enumerate(result["hits"], start=1):
        source_units = hit.get("sourceUnits") or []
        evidence.append(
            {
                "evidenceId": f"ev-{i}",
                "kbId": hit.get("kbId"),
                "documentId": hit.get("documentId"),
                "versionId": hit.get("versionId"),
                "buildId": hit.get("buildId"),
                "chunkId": hit.get("chunkId"),
                "parentChunkId": hit.get("parentChunkId"),
                "unitId": source_units[0].get("unitId") if source_units else None,
                "location": source_units[0].get("location") if source_units else {},
                "text": hit.get("text"),
                "score": round(float(hit.get("score", 0.0)), 6),
                "rerankScore": hit.get("rerankScore"),
            }
        )

    return {
        "query": query,
        "configId": result["config"]["config_id"],
        "warnings": result["warnings"],
        "evidence": evidence,
    }
