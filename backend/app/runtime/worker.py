"""持久化后台任务 Worker。

通过数据库 job 表实现可恢复的异步任务执行，而不是依赖进程内线程：
- claim 时加行锁并递增 lease_epoch；
- heartbeat 延长租约；
- lease 过期后其他 worker 可接管；
- 失败只对“可重试错误”重新入队；
- 所有写回都带当前 lease_epoch，防止旧 worker 晚到后覆盖新 worker 结果。

这套 lease_epoch 机制是防止“僵尸 worker 写回”的关键。
"""
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import and_, or_, select, update

from app.db import get_db_connection
from app.errors import ApiException
from app.knowledge.artifacts import read_artifact
from app.models import app_user_table, deletion_record_table, job_table, qa_run_table
from app.qa.runtime import is_retryable_job_code
from app.logging_config import configure_logging, log_context
from app.telemetry import configure_telemetry, observation_batch, shutdown_telemetry

logger = logging.getLogger(__name__)


class DurableJobWorker:
    """从 job 表竞争任务并执行 INGEST / PURGE / EVALUATE / QA_RUN。"""
    def __init__(
        self,
        worker_id: Optional[str] = None,
        lease_seconds: int = 120,
        poll_seconds: float = 1.0,
        max_attempts: int = 3,
        kinds: Optional[set[str]] = None,
    ):
        # 默认 worker_id 由主机名 + PID 组成，便于日志定位具体执行进程。
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.max_attempts = max_attempts
        self.kinds = kinds or {"INGEST", "PURGE", "EVALUATE", "QA_RUN"}
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run_forever(self):
        """持续轮询；当前没有任务时按 poll_seconds 休眠，避免空转占 CPU。"""
        configure_telemetry()
        logger.info("Durable worker started: %s", self.worker_id)
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                logger.exception("Durable worker iteration failed")
                worked = False
            if not worked:
                self._stop.wait(self.poll_seconds)

    def run_once(self) -> bool:
        """竞争一个任务并完整执行一次；返回 False 表示当前无可执行任务。"""
        job = self._claim_one()
        if not job:
            return False
        with log_context(tenant_id=str(job["tenant_id"]), job_id=str(job["id"]),
                         run_id=str(job["target_id"]) if job["kind"] == "QA_RUN" else None,
                         worker_id=self.worker_id, kind=job["kind"], lease_epoch=job["lease_epoch"],
                         attempt=int(job.get("attempt") or 0) + 1):
            with observation_batch(job):
                self._execute_claimed(job)
        return True

    def _execute_claimed(self, job):
        started = time.perf_counter()
        logger.info("job_started")
        heartbeat_stop = threading.Event()
        # 业务执行可能持续数十秒/数分钟，因此独立 heartbeat 线程持续续租。
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job, heartbeat_stop),
            name=f"job-heartbeat-{job['id']}",
            daemon=True,
        )
        heartbeat.start()
        try:
            if job["kind"] == "INGEST":
                self._run_ingest(job)
            elif job["kind"] == "PURGE":
                self._run_purge(job)
            elif job["kind"] == "EVALUATE":
                self._run_evaluate(job)
            elif job["kind"] == "QA_RUN":
                self._run_qa(job)
            else:
                self._fail(job, "UNSUPPORTED_JOB_KIND")
        except Exception as exc:
            logger.exception("Job %s failed", job["id"])
            code = exc.code if isinstance(exc, ApiException) else "WORKER_EXECUTION_FAILED"
            self._fail(job, code)
        finally:
            # 无论业务成功/失败都停止 heartbeat，避免任务结束后继续续租。
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
        # 业务执行统一先落 FAILED，再由这里根据错误类型决定是否重新 QUEUED。
        self._requeue_failed_if_retryable(job)
        try:
            with get_db_connection() as conn:
                row = conn.execute(select(job_table.c.status, job_table.c.stage, job_table.c.error_code)
                    .where(and_(job_table.c.tenant_id == job["tenant_id"], job_table.c.id == job["id"],
                                job_table.c.lease_epoch == job["lease_epoch"]))).first()
            logger.log(logging.WARNING if row and row.error_code else logging.INFO,
                       "job_finished" if row else "job_lease_lost", extra={"fields": {
                           "status": str(row.status) if row else "LEASE_LOST",
                           "stage": str(row.stage) if row else None,
                           "error_code": row.error_code if row else None,
                           "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                       }})
        except Exception:
            logger.exception("job_status_log_failed")

    def _claim_one(self) -> Optional[Dict[str, Any]]:
        """用 FOR UPDATE SKIP LOCKED 原子竞争一个可执行/租约过期任务。"""
        now = datetime.now(timezone.utc)
        with get_db_connection() as conn:
            row = conn.execute(
                select(job_table)
                .where(and_(
                    job_table.c.kind.in_(sorted(self.kinds)),
                    job_table.c.available_at <= now,
                    or_(job_table.c.status == "QUEUED",
                        and_(job_table.c.status == "RUNNING", job_table.c.lease_until.isnot(None), job_table.c.lease_until < now)),
                ))
                .order_by(job_table.c.created_at.asc())
                .limit(1)
                # 多 worker 并发时，已被其他事务锁住的任务直接跳过，减少互相等待。
                .with_for_update(skip_locked=True)
            ).first()
            if not row:
                return None
            job = dict(row._mapping)
            # 每次重新 claim 都递增 epoch；旧 worker 后续写回若带旧 epoch 会被 WHERE 条件拒绝。
            next_epoch = int(job.get("lease_epoch") or 0) + 1
            conn.execute(
                update(job_table)
                .where(and_(job_table.c.tenant_id == job["tenant_id"], job_table.c.id == job["id"]))
                .values(status="RUNNING", lease_until=now + timedelta(seconds=self.lease_seconds),
                        lease_epoch=next_epoch, attempt=job_table.c.attempt + 1,
                        error_code=None, updated_at=now)
            )
            conn.commit()
            job["lease_epoch"] = next_epoch
            return job

    def _heartbeat_loop(self, job: Dict[str, Any], stop: threading.Event):
        """周期延长 lease；rowcount=0 代表租约已丢失，立即停止续租。"""
        # 在租约 1/3 周期续租，给数据库短暂抖动留出余量。
        interval = max(5.0, self.lease_seconds / 3.0)
        while not stop.wait(interval):
            now = datetime.now(timezone.utc)
            with get_db_connection() as conn:
                result = conn.execute(
                    update(job_table)
                    .where(and_(
                        job_table.c.tenant_id == job["tenant_id"],
                        job_table.c.id == job["id"],
                        job_table.c.status == "RUNNING",
                        job_table.c.lease_epoch == job["lease_epoch"],
                    ))
                    .values(
                        lease_until=now + timedelta(seconds=self.lease_seconds),
                        updated_at=now,
                    )
                )
                conn.commit()
                if result.rowcount == 0:
                    logger.warning("job_lease_lost", extra={"fields": {
                        "tenant_id": str(job["tenant_id"]), "job_id": str(job["id"]),
                        "worker_id": self.worker_id, "lease_epoch": job["lease_epoch"],
                    }})
                    return

    def _run_ingest(self, job: Dict[str, Any]):
        from app.knowledge.service import KnowledgeService
        cp = job.get("checkpoint") or {}
        data = read_artifact(str(cp["storageKey"]))
        KnowledgeService._execute_ingestion_job(
            str(job["tenant_id"]), str(job["id"]), str(cp["documentId"]),
            str(cp["versionId"]), str(cp["buildId"]), data, str(cp["filename"]),
            lease_epoch=int(job["lease_epoch"]),
        )

    def _run_purge(self, job: Dict[str, Any]):
        from app.knowledge.service import KnowledgeService
        cp = job.get("checkpoint") or {}
        KnowledgeService._execute_purge(
            str(job["tenant_id"]), str(job["target_id"]), str(job["id"]), str(cp["deletionRecordId"]),
            lease_epoch=int(job["lease_epoch"]),
        )

    def _run_evaluate(self, job: Dict[str, Any]):
        """按任务创建者重新构建 Actor；用户失效后不能继续跑评估。"""
        from app.evaluation.runner import EvaluationRunner
        from app.identity.deps import Actor

        with get_db_connection() as conn:
            user = conn.execute(
                select(
                    app_user_table.c.id,
                    app_user_table.c.is_admin,
                    app_user_table.c.status,
                ).where(and_(
                    app_user_table.c.tenant_id == job["tenant_id"],
                    app_user_table.c.id == job["requested_by"],
                ))
            ).first()
        # 创建任务时有权限，不代表执行时仍有效；异步边界必须重新确认用户状态。
        if not user or str(user.status) != "ACTIVE":
            self._fail(job, "REQUESTER_NOT_ACTIVE")
            return

        actor = Actor(
            tenant_id=str(job["tenant_id"]),
            user_id=str(user.id),
            admin=bool(user.is_admin),
            security_epoch=0,
        )
        if (job.get("checkpoint") or {}).get("diagnosticType") == "independent-assertions-v1":
            from app.evaluation.assertion_diagnostic import execute_diagnostic
            execute_diagnostic(actor, job)
        else:
            EvaluationRunner._execute_run(actor, str(job["target_id"]))

    def _run_qa(self, job: Dict[str, Any]):
        """执行排队中的 QA run，并把 job_id / lease_epoch 传到底层用于防旧 worker 写回。"""
        from app.identity.deps import Actor
        from app.qa.service import QaService

        with get_db_connection() as conn:
            user = conn.execute(
                select(
                    app_user_table.c.id,
                    app_user_table.c.is_admin,
                    app_user_table.c.status,
                ).where(and_(
                    app_user_table.c.tenant_id == job["tenant_id"],
                    app_user_table.c.id == job["requested_by"],
                ))
            ).first()
        if not user or str(user.status) != "ACTIVE":
            self._fail(job, "REQUESTER_NOT_ACTIVE")
            return

        actor = Actor(
            tenant_id=str(job["tenant_id"]),
            user_id=str(user.id),
            admin=bool(user.is_admin),
            security_epoch=0,
        )
        QaService.execute_queued_run(
            actor,
            str(job["target_id"]),
            job_id=str(job["id"]),
            lease_epoch=int(job["lease_epoch"]),
        )

    def _requeue_failed_if_retryable(self, job: Dict[str, Any]):
        """仅对临时性失败进行指数退避后重新入队。"""
        with get_db_connection() as conn:
            row = conn.execute(
                select(
                    job_table.c.status,
                    job_table.c.attempt,
                    job_table.c.lease_epoch,
                    job_table.c.error_code,
                ).where(and_(
                    job_table.c.tenant_id == job["tenant_id"],
                    job_table.c.id == job["id"],
                ))
            ).first()
            if not row or str(row.status) != "FAILED":
                return
            # 达到最大尝试次数、确定性错误、或 lease 已换代时都禁止重试。
            if int(row.attempt or 0) >= self.max_attempts:
                return
            if not is_retryable_job_code(str(row.error_code or "")):
                return
            if int(row.lease_epoch or 0) != int(job.get("lease_epoch") or 0):
                return
            # 2,4,8... 秒指数退避，上限 60 秒，避免故障时持续热循环。
            delay_seconds = min(60, 2 ** max(0, int(row.attempt or 1) - 1))
            now = datetime.now(timezone.utc)
            conn.execute(
                update(job_table)
                .where(and_(
                    job_table.c.tenant_id == job["tenant_id"],
                    job_table.c.id == job["id"],
                    job_table.c.status == "FAILED",
                    job_table.c.lease_epoch == job["lease_epoch"],
                ))
                .values(
                    status="QUEUED",
                    available_at=now + timedelta(seconds=delay_seconds),
                    lease_until=None,
                    updated_at=now,
                )
            )
            # job 重试时同步把 QA run 从 FAILED 拉回 QUEUED，保持两张表状态一致。
            if job.get("kind") == "QA_RUN":
                conn.execute(
                    update(qa_run_table)
                    .where(and_(
                        qa_run_table.c.tenant_id == job["tenant_id"],
                        qa_run_table.c.id == job["target_id"],
                        qa_run_table.c.status == "FAILED",
                    ))
                    .values(
                        status="QUEUED",
                        stage="ROUTING",
                        terminal_reason=None,
                        finished_at=None,
                        updated_at=now,
                    )
                )
            conn.commit()

            logger.warning("job_retry_scheduled", extra={"fields": {
                "error_code": row.error_code, "retry_delay_seconds": delay_seconds,
            }})

    def _fail(self, job: Dict[str, Any], code: str):
        """将当前 lease 对应的任务标记失败；旧 lease 无法修改新 worker 的任务。"""
        now = datetime.now(timezone.utc)
        with get_db_connection() as conn:
            conn.execute(
                update(job_table)
                .where(and_(
                    job_table.c.tenant_id == job["tenant_id"],
                    job_table.c.id == job["id"],
                    job_table.c.lease_epoch == job.get("lease_epoch"),
                ))
                .values(status="FAILED", error_code=code, lease_until=None, updated_at=now)
            )
            if job.get("kind") == "PURGE":
                deletion_record_id = (job.get("checkpoint") or {}).get("deletionRecordId")
                if deletion_record_id:
                    conn.execute(
                        update(deletion_record_table)
                        .where(and_(
                            deletion_record_table.c.tenant_id == job["tenant_id"],
                            deletion_record_table.c.id == deletion_record_id,
                        ))
                        .values(status="FAILED", updated_at=now)
                    )
            conn.commit()


def main():
    configure_logging()
    try:
        DurableJobWorker().run_forever()
    finally:
        shutdown_telemetry(float(os.getenv("RAG_LANGFUSE_SHUTDOWN_TIMEOUT_SECONDS", "2")))


if __name__ == "__main__":
    main()
