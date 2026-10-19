from contextlib import asynccontextmanager
import logging
import threading
import time
import uuid
from urllib.parse import urlparse
from fastapi import FastAPI, Request, Depends, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, and_, text

from app.config import settings, is_test_env
from app.db import get_db_connection
from app.errors import ApiException, api_exception_handler
from app.identity.deps import Actor, get_current_actor
from app.models import tenant_table, app_user_table, user_group_table, group_member_table
from app.logging_config import configure_logging, log_context

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not is_test_env():
        for name in ("RAG_CSRF_SECRET", "LITELLM_MASTER_KEY"):
            if not getattr(settings, name, "").strip():
                raise RuntimeError(f"{name} is required; initialize the environment first")
    configure_logging()
    from app.telemetry import configure_telemetry
    configure_telemetry()

    # 启动阶段先确认数据库可连接并执行幂等 seed；尽早暴露基础配置问题，避免首个业务请求才失败。
    try:
        with get_db_connection() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        if not is_test_env():
            raise RuntimeError(f"Fatal: Database connection failed during startup: {e}") from e
        print(f"Warning: Database startup check failed: {e}")

    try:
        from app.seed import DemoSeed
        DemoSeed.run()
    except Exception as e:
        if not is_test_env():
            raise RuntimeError(f"Fatal: Database seed failed during startup: {e}") from e
        print(f"Warning: Database seed check: {e}")
    try:
        from app.qa.service import QaService
        QaService.recover_unfinished_runs()
    except Exception as e:
        print(f"Warning: Unfinished run recovery check: {e}")
    try:
        from app.knowledge.service import KnowledgeService
        KnowledgeService.recover_unfinished_jobs()
    except Exception as e:
        print(f"Warning: Unfinished job recovery check: {e}")

    workers = []
    worker_threads = []
    if not is_test_env() and settings.RAG_EMBEDDED_WORKER:
        from app.runtime.worker import DurableJobWorker

        pools = [
            ("qa", {"QA_RUN"}, max(1, settings.RAG_QA_WORKERS)),
            ("ingest", {"INGEST", "PURGE"}, max(1, settings.RAG_INGEST_WORKERS)),
            ("eval", {"EVALUATE"}, max(1, settings.RAG_EVAL_WORKERS)),
        ]
        for pool_name, kinds, count in pools:
            for index in range(count):
                worker = DurableJobWorker(
                    worker_id=f"embedded-{pool_name}-{index}",
                    kinds=kinds,
                )
                thread = threading.Thread(
                    target=worker.run_forever,
                    name=f"rag-worker-{pool_name}-{index}",
                    daemon=True,
                )
                thread.start()
                workers.append(worker)
                worker_threads.append(thread)

    yield

    for worker in workers:
        worker.stop()
    for thread in worker_threads:
        thread.join(timeout=5)
    from app.telemetry import shutdown_telemetry
    shutdown_telemetry(float(__import__("os").getenv("RAG_LANGFUSE_SHUTDOWN_TIMEOUT_SECONDS", "2")))


app = FastAPI(
    title="RAG Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(ApiException, api_exception_handler)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    # CSRF 只保护 API 中会修改状态的方法；GET/HEAD 保持只读，不要求携带 CSRF token。
    exempt_paths = {"/api/v1/auth/csrf", "/api/v1/health", "/health", "/docs", "/openapi.json"}
    internal_service_call = request.url.path.startswith("/api/v1/internal/")
    if request.method in ("POST", "PUT", "DELETE", "PATCH") and request.url.path.startswith("/api/v1"):
        if request.url.path not in exempt_paths and not internal_service_call:
            # 第一步校验 Origin（若浏览器提供）：非可信来源直接拒绝，先阻断跨站写请求。
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                req_host = request.url.hostname
                req_port = request.url.port or (443 if request.url.scheme == "https" else 80)
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                if parsed.hostname != req_host or origin_port != req_port or parsed.scheme != request.url.scheme:
                    return JSONResponse(status_code=403, content={"code": "CSRF_INVALID", "message": "请刷新安全凭据后重试"})

            # 第二步同时读取 Header 与 Cookie 中的 CSRF token；双提交值必须同时存在并匹配。
            csrf_header = request.headers.get("x-csrf-token")
            csrf_cookie = request.cookies.get("RAG_CSRF")
            if not csrf_header or not csrf_cookie or csrf_header != csrf_cookie:
                return JSONResponse(status_code=403, content={"code": "CSRF_INVALID", "message": "请刷新安全凭据后重试"})

            # 第三步确定 token 的会话绑定关系；登录态 token 不能被匿名态或其他会话复用。
            session_token = request.cookies.get("RAG_SESSION")
            binding = session_token if session_token else request.cookies.get("RAG_NONCE", "")
            if not verify_csrf_token(csrf_header, binding, settings.RAG_CSRF_SECRET):
                return JSONResponse(status_code=403, content={"code": "CSRF_INVALID", "message": "请刷新安全凭据后重试"})

    response = await call_next(request)
    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.scope["request_id"] = request_id
    started = time.perf_counter()
    with log_context(request_id=request_id):
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", extra={"fields": {
                "method": request.method,
                "route": getattr(request.scope.get("route"), "path", "<unmatched>"),
                "status": 500, "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }})
            raise
        response.headers["X-Request-ID"] = request_id
        logger.log(logging.WARNING if response.status_code >= 400 else logging.INFO,
                   "request_finished", extra={"fields": {
                       "method": request.method,
                       "route": getattr(request.scope.get("route"), "path", "<unmatched>"),
                       "status": response.status_code,
                       "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                   }})
        return response


from app.identity.router import router as auth_router
from app.identity.admin_router import router as identity_admin_router
from app.knowledge.router import router as knowledge_router
from app.retrieval.router import router as retrieval_router
from app.qa.router import router as qa_router
from app.admin.config_router import router as config_router
from app.evaluation.router import router as eval_router
from app.model_budget_router import router as budget_router
from app.identity.security import verify_csrf_token

# 统一注册各业务路由；安全中间件放在应用层统一生效，避免某个 router 漏掉保护。
app.include_router(auth_router, prefix="/api/v1")
app.include_router(identity_admin_router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(eval_router, prefix="/api/v1")
app.include_router(budget_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "服务器内部错误，请稍后重试"},
        headers={"X-Request-ID": request.scope.get("request_id", "")},
    )


@app.get("/api/v1/me")
def me(actor: Actor = Depends(get_current_actor)):
    with get_db_connection() as conn:
        user_stmt = select(app_user_table.c.login_name).where(
            and_(
                app_user_table.c.tenant_id == actor.tenant_id,
                app_user_table.c.id == actor.user_id,
            )
        )
        user = conn.execute(user_stmt).first()
        if not user:
            raise ApiException(401, "AUTH_REQUIRED", "用户不存在")

        tenant_row = conn.execute(
            select(tenant_table.c.code, tenant_table.c.security_epoch).where(
                tenant_table.c.id == actor.tenant_id
            )
        ).first()
        epoch = int(tenant_row.security_epoch) if tenant_row else 0
        tenant_code = str(tenant_row.code) if tenant_row else ""

        groups_stmt = (
            select(user_group_table.c.id, user_group_table.c.name)
            .select_from(
                user_group_table.join(
                    group_member_table,
                    and_(
                        group_member_table.c.tenant_id == user_group_table.c.tenant_id,
                        group_member_table.c.group_id == user_group_table.c.id,
                    ),
                )
            )
            .where(
                and_(
                    group_member_table.c.tenant_id == actor.tenant_id,
                    group_member_table.c.user_id == actor.user_id,
                )
            )
        )
        groups = conn.execute(groups_stmt).fetchall()

        return {
            "id": actor.user_id,
            "userId": actor.user_id,
            "tenantId": actor.tenant_id,
            "tenantCode": tenant_code,
            "loginName": user.login_name,
            "admin": actor.admin,
            "isAdmin": actor.admin,
            "securityEpoch": epoch,
            "groups": [{"id": str(g.id), "name": str(g.name)} for g in groups],
        }


@app.get("/api/v1/health")
@app.get("/health")
def health():
    with get_db_connection() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "db": "connected"}
