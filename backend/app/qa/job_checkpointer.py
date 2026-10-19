import base64
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from sqlalchemy import and_, select, update

from app.db import get_db_connection
from app.models import job_table


class JobCheckpointSaver(BaseCheckpointSaver):
    """把 LangGraph checkpoint 存进现有 QA_RUN job.checkpoint JSON。

    这里直接实现 LangGraph 官方 Checkpointer 协议，而不是另造一套恢复机制。
    checkpoint 写入与 DurableJob 使用同一个 lease_epoch fencing：
    旧 worker 即使在进程恢复后“晚到”，也无法覆盖新 worker 已经推进的图状态。
    """

    _ROOT_KEY = "langgraph"
    # 限制单个 run 的 checkpoint 历史数量，防止长 Agent 运行无限膨胀 job JSON。
    _MAX_HISTORY = 64

    @staticmethod
    def _cfg(
        config: RunnableConfig,
    ) -> tuple[str, str, str, Optional[str], Optional[str], Optional[int]]:
        # thread_id 对应 qa_run_id；job_id + lease_epoch 只在真实 Worker 执行时携带。
        values = config.get("configurable", {})
        thread_id = str(values["thread_id"])
        tenant_id = str(values.get("tenant_id") or "")
        checkpoint_ns = str(values.get("checkpoint_ns") or "")
        checkpoint_id = values.get("checkpoint_id")
        job_id = values.get("job_id")
        lease_epoch = values.get("lease_epoch")
        return (
            tenant_id,
            thread_id,
            checkpoint_ns,
            str(checkpoint_id) if checkpoint_id else None,
            str(job_id) if job_id else None,
            int(lease_epoch) if lease_epoch is not None else None,
        )

    def _encode(self, value: Any) -> Dict[str, str]:
        kind, payload = self.serde.dumps_typed(value)
        return {
            "type": kind,
            "data": base64.b64encode(payload).decode("ascii"),
        }

    def _decode(self, value: Dict[str, str]) -> Any:
        return self.serde.loads_typed(
            (value["type"], base64.b64decode(value["data"].encode("ascii")))
        )

    @staticmethod
    def _empty_root() -> Dict[str, Any]:
        return {"checkpoints": [], "writes": {}}

    def _load_job_for_update(
        self,
        conn,
        tenant_id: str,
        thread_id: str,
        job_id: Optional[str] = None,
        lease_epoch: Optional[int] = None,
    ):
        clauses = [
            job_table.c.kind == "QA_RUN",
            job_table.c.target_id == thread_id,
        ]
        if tenant_id:
            clauses.append(job_table.c.tenant_id == tenant_id)
        # 有 worker lease 时必须严格匹配 RUNNING + epoch；
        # HITL WAITING_APPROVAL 等脱离 worker 的读操作则走较宽松状态集合。
        if job_id is not None and lease_epoch is not None:
            clauses.extend([
                job_table.c.id == job_id,
                job_table.c.status == "RUNNING",
                job_table.c.lease_epoch == lease_epoch,
            ])
        else:
            clauses.append(job_table.c.status.in_(["QUEUED", "RUNNING", "WAITING_APPROVAL"]))
        stmt = select(
            job_table.c.id,
            job_table.c.tenant_id,
            job_table.c.checkpoint,
            job_table.c.status,
            job_table.c.lease_epoch,
        ).where(and_(*clauses))
        return conn.execute(stmt.with_for_update()).first()

    def _load_job(self, conn, tenant_id: str, thread_id: str):
        stmt = select(job_table.c.id, job_table.c.tenant_id, job_table.c.checkpoint).where(
            and_(
                job_table.c.kind == "QA_RUN",
                job_table.c.target_id == thread_id,
            )
        )
        if tenant_id:
            stmt = stmt.where(job_table.c.tenant_id == tenant_id)
        return conn.execute(stmt).first()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tenant_id, thread_id, checkpoint_ns, checkpoint_id, job_id, lease_epoch = self._cfg(config)
        with get_db_connection() as conn:
            row = self._load_job(conn, tenant_id, thread_id)
        if not row:
            return None

        job_cp = dict(row.checkpoint or {})
        root = dict(job_cp.get(self._ROOT_KEY) or self._empty_root())
        # 同一 namespace 可以有多个 checkpoint；无指定 checkpoint_id 时读取最新一个。
        checkpoints = [
            cp
            for cp in root.get("checkpoints", [])
            if str(cp.get("ns") or "") == checkpoint_ns
        ]
        if checkpoint_id:
            match = next((cp for cp in checkpoints if cp.get("id") == checkpoint_id), None)
        else:
            match = checkpoints[-1] if checkpoints else None
        if not match:
            return None

        cp_id = str(match["id"])
        writes_key = f"{checkpoint_ns}:{cp_id}"
        # pending_writes 是 LangGraph 节点尚未并入 checkpoint 的 channel 写入，恢复时必须一并返回。
        pending = []
        for item in root.get("writes", {}).get(writes_key, []):
            pending.append(
                (
                    str(item["taskId"]),
                    str(item["channel"]),
                    self._decode(item["value"]),
                )
            )

        effective_tenant = tenant_id or str(row.tenant_id)
        saved_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": effective_tenant,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
                **({"job_id": job_id, "lease_epoch": lease_epoch} if job_id is not None and lease_epoch is not None else {}),
            }
        }
        parent_id = match.get("parentId")
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "tenant_id": effective_tenant,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )

        return CheckpointTuple(
            config=saved_config,
            checkpoint=self._decode(match["checkpoint"]),
            metadata=self._decode(match["metadata"]),
            parent_config=parent_config,
            pending_writes=pending,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        tenant_id, thread_id, checkpoint_ns, _, _, _ = self._cfg(config)
        with get_db_connection() as conn:
            row = self._load_job(conn, tenant_id, thread_id)
        if not row:
            return

        root = dict((row.checkpoint or {}).get(self._ROOT_KEY) or self._empty_root())
        before_id = None
        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")

        emitted = 0
        for item in reversed(root.get("checkpoints", [])):
            if str(item.get("ns") or "") != checkpoint_ns:
                continue
            if before_id and str(item.get("id")) >= str(before_id):
                continue
            metadata = self._decode(item["metadata"])
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue
            cp_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "tenant_id": tenant_id or str(row.tenant_id),
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": str(item["id"]),
                }
            }
            result = self.get_tuple(cp_config)
            if result is not None:
                yield result
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        del new_versions
        tenant_id, thread_id, checkpoint_ns, parent_id, job_id, lease_epoch = self._cfg(config)
        now = datetime.now(timezone.utc)
        cp_id = str(checkpoint["id"])

        with get_db_connection() as conn:
            row = self._load_job_for_update(conn, tenant_id, thread_id, job_id, lease_epoch)
            if not row:
                raise RuntimeError(f"QA_RUN job not found for thread {thread_id}")

            job_cp = dict(row.checkpoint or {})
            root = dict(job_cp.get(self._ROOT_KEY) or self._empty_root())
            # put 具备幂等性：相同 ns + checkpoint_id 先移除旧项再追加最新值。
            checkpoints = [
                cp for cp in root.get("checkpoints", [])
                if not (cp.get("id") == cp_id and str(cp.get("ns") or "") == checkpoint_ns)
            ]
            checkpoints.append(
                {
                    "id": cp_id,
                    "ns": checkpoint_ns,
                    "parentId": parent_id,
                    "checkpoint": self._encode(checkpoint),
                    "metadata": self._encode(metadata),
                    "createdAt": now.isoformat(),
                }
            )
            # 只保留最近历史；LangGraph 正常恢复只需要最新链路，不需要无限保存。
            if len(checkpoints) > self._MAX_HISTORY:
                checkpoints = checkpoints[-self._MAX_HISTORY:]
            root["checkpoints"] = checkpoints
            job_cp[self._ROOT_KEY] = root

            clauses = [job_table.c.id == row.id]
            if job_id is not None and lease_epoch is not None:
                clauses.extend([
                    job_table.c.status == "RUNNING",
                    job_table.c.lease_epoch == lease_epoch,
                ])
            else:
                clauses.append(job_table.c.status.in_(["QUEUED", "RUNNING", "WAITING_APPROVAL"]))
            result = conn.execute(
                update(job_table)
                .where(and_(*clauses))
                .values(checkpoint=job_cp, updated_at=now)
            )
            # 写入失败通常表示 lease 已换代：旧 worker 必须放弃当前 checkpoint。
            if result.rowcount != 1:
                conn.rollback()
                raise RuntimeError(f"QA_RUN lease lost for checkpoint thread {thread_id}")
            conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
                **({"job_id": job_id, "lease_epoch": lease_epoch} if job_id is not None and lease_epoch is not None else {}),
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        tenant_id, thread_id, checkpoint_ns, checkpoint_id, job_id, lease_epoch = self._cfg(config)
        if not checkpoint_id:
            return
        now = datetime.now(timezone.utc)
        key = f"{checkpoint_ns}:{checkpoint_id}"

        with get_db_connection() as conn:
            row = self._load_job_for_update(conn, tenant_id, thread_id, job_id, lease_epoch)
            if not row:
                return
            job_cp = dict(row.checkpoint or {})
            root = dict(job_cp.get(self._ROOT_KEY) or self._empty_root())
            all_writes = dict(root.get("writes") or {})
            existing = list(all_writes.get(key) or [])
            # 同一 taskId + write index 重放时覆盖原值，保证节点重执行不会重复追加 pending write。
            by_identity = {
                (str(item.get("taskId")), int(item.get("idx", 0))): item
                for item in existing
            }
            for idx, (channel, value) in enumerate(writes):
                by_identity[(task_id, idx)] = {
                    "taskId": task_id,
                    "taskPath": task_path,
                    "idx": idx,
                    "channel": channel,
                    "value": self._encode(value),
                }
            all_writes[key] = list(by_identity.values())
            root["writes"] = all_writes
            job_cp[self._ROOT_KEY] = root
            clauses = [job_table.c.id == row.id]
            if job_id is not None and lease_epoch is not None:
                clauses.extend([
                    job_table.c.status == "RUNNING",
                    job_table.c.lease_epoch == lease_epoch,
                ])
            else:
                clauses.append(job_table.c.status.in_(["QUEUED", "RUNNING", "WAITING_APPROVAL"]))
            result = conn.execute(
                update(job_table)
                .where(and_(*clauses))
                .values(checkpoint=job_cp, updated_at=now)
            )
            # pending write 同样受 lease fencing；旧 worker 的写入静默回滚。
            if result.rowcount == 1:
                conn.commit()
            else:
                conn.rollback()

    def delete_thread(self, thread_id: str) -> None:
        """只删除 LangGraph 子树，不影响 job 里其他业务 checkpoint 字段。"""
        now = datetime.now(timezone.utc)
        with get_db_connection() as conn:
            rows = conn.execute(
                select(job_table.c.id, job_table.c.checkpoint).where(
                    and_(
                        job_table.c.kind == "QA_RUN",
                        job_table.c.target_id == thread_id,
                    )
                )
            ).fetchall()
            for row in rows:
                job_cp = dict(row.checkpoint or {})
                job_cp.pop(self._ROOT_KEY, None)
                conn.execute(
                    update(job_table)
                    .where(job_table.c.id == row.id)
                    .values(checkpoint=job_cp, updated_at=now)
                )
            conn.commit()
