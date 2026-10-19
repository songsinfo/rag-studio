from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

from sqlalchemy import and_, select

from app.models import job_table

_execution_lease: ContextVar[Optional[Tuple[str, int]]] = ContextVar("qa_execution_lease", default=None)

@contextmanager
def qa_execution_scope(job_id: Optional[str], lease_epoch: Optional[int]) -> Iterator[None]:
    if not job_id or lease_epoch is None:
        yield
        return
    token = _execution_lease.set((str(job_id), int(lease_epoch)))
    try:
        yield
    finally:
        _execution_lease.reset(token)

def current_qa_execution() -> Optional[Tuple[str, int]]:
    return _execution_lease.get()

def qa_lease_is_current(conn, tenant_id: str, run_id: str) -> bool:
    current = current_qa_execution()
    if current is None:
        return True
    job_id, lease_epoch = current
    now = datetime.now(timezone.utc)
    row = conn.execute(
        select(job_table.c.id)
        .where(and_(
            job_table.c.tenant_id == tenant_id,
            job_table.c.id == job_id,
            job_table.c.kind == "QA_RUN",
            job_table.c.target_id == run_id,
            job_table.c.status == "RUNNING",
            job_table.c.lease_epoch == lease_epoch,
            job_table.c.lease_until > now,
        ))
        .with_for_update()
    ).first()
    return row is not None
