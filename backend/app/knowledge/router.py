import hmac
import json
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Header, UploadFile, File, Form, status, Response, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.identity.deps import Actor, get_current_actor, require_admin
from app.errors import ApiException
from app.knowledge.service import KnowledgeService
from app.knowledge.imports import create_import_preview, consume_import_preview, split_wiki_package

router = APIRouter(tags=["knowledge"])


class CreateKbRequest(BaseModel):
    name: str
    readers: List[Dict[str, Any]] = []


class UpdateKbAclRequest(BaseModel):
    expectedVersion: int
    readers: List[Dict[str, Any]] = []


class UpdateDocAclRequest(BaseModel):
    expectedVersion: int
    mode: str = "RESTRICT"
    readers: List[Dict[str, Any]] = []


class LifecycleActionRequest(BaseModel):
    expectedVersion: int



# 知识库管理接口：负责创建、列表、详情以及知识库级权限/配置入口。
@router.get("/knowledge-bases")
def list_kbs(actor: Actor = Depends(get_current_actor)):
    return KnowledgeService.list_kbs(actor)


@router.post("/admin/knowledge-bases")
def create_kb(payload: CreateKbRequest, actor: Actor = Depends(require_admin)):
    return KnowledgeService.create_kb(actor, payload.name, payload.readers)


@router.get("/admin/knowledge-bases/{kb_id}/acl")
def get_kb_acl(kb_id: str, actor: Actor = Depends(require_admin)):
    return KnowledgeService.get_kb_acl(actor, kb_id)


@router.put("/admin/knowledge-bases/{kb_id}/acl")
def update_kb_acl(kb_id: str, payload: UpdateKbAclRequest, actor: Actor = Depends(require_admin)):
    return KnowledgeService.update_kb_acl(actor, kb_id, payload.expectedVersion, payload.readers)


# 文档接口：负责上传、查询和版本操作；解析、切分和索引由后台 durable job 异步完成。
@router.get("/documents")
def list_documents(kbId: Optional[str] = None, actor: Actor = Depends(get_current_actor)):
    return KnowledgeService.list_documents(actor, kb_id=kbId, admin_mode=False)


@router.get("/admin/documents")
def list_admin_documents(
    kbId: Optional[str] = None,
    status: str = "LIVE",
    actor: Actor = Depends(require_admin),
):
    return KnowledgeService.list_documents(actor, kb_id=kbId, admin_mode=True, status=status)


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, actor: Actor = Depends(get_current_actor)):
    return KnowledgeService.get_document(actor, doc_id)


@router.post("/admin/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    kbId: str = Form(...),
    buildConfig: str = Form(default="{}"),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: Actor = Depends(require_admin),
):
    content = await file.read(26214401)
    try:
        cfg = json.loads(buildConfig) if buildConfig else {}
    except Exception as exc:
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是合法 JSON") from exc
    if not isinstance(cfg, dict):
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是 JSON 对象")
    return await run_in_threadpool(KnowledgeService.upload_document,
        actor=actor,
        kb_id=kbId,
        document_id=None,
        filename=file.filename or "uploaded.txt",
        file_bytes=content,
        idempotency_key=idempotency_key,
        build_config=cfg,
    )


@router.post("/admin/documents/batch")
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    kbId: str = Form(...),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: Actor = Depends(require_admin),
):
    if not files or len(files) > 100:
        raise ApiException(413, "INPUT_TOO_LARGE", "批量上传最多100个文件")
    results = []
    for index, file in enumerate(files):
        try:
            results.append(
                KnowledgeService.upload_document(
                    actor=actor,
                    kb_id=kbId,
                    document_id=None,
                    filename=file.filename or f"uploaded-{index}.txt",
                    file_bytes=await file.read(),
                    idempotency_key=f"{idempotency_key}-{index}",
                    build_config={},
                )
            )
        except ApiException as exc:
            results.append(
                {
                    "fileName": file.filename or f"uploaded-{index}.txt",
                    "status": "FAILED",
                    "errorCode": exc.code,
                }
            )
    return results


@router.post("/admin/documents/{doc_id}/versions", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_version(
    doc_id: str,
    file: UploadFile = File(...),
    buildConfig: str = Form(default="{}"),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: Actor = Depends(require_admin),
):
    doc = await run_in_threadpool(KnowledgeService.get_document, actor, doc_id)
    content = await file.read(26214401)
    try:
        cfg = json.loads(buildConfig) if buildConfig else {}
    except Exception as exc:
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是合法 JSON") from exc
    if not isinstance(cfg, dict):
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是 JSON 对象")
    return await run_in_threadpool(KnowledgeService.upload_document,
        actor=actor,
        kb_id=doc["kbId"],
        document_id=doc_id,
        filename=file.filename or "uploaded.txt",
        file_bytes=content,
        idempotency_key=idempotency_key,
        build_config=cfg,
    )


@router.post("/admin/import-previews")
def preview_directory_import(
    body: Dict[str, Any],
    actor: Actor = Depends(require_admin),
):
    root_alias = str(body.get("rootAlias") or "")
    if root_alias and root_alias != "default":
        raise ApiException(400, "INVALID_INPUT", "rootAlias 无效")
    return create_import_preview(
        actor.tenant_id,
        actor.user_id,
        str(body.get("relativePath") or ""),
    )


@router.post("/admin/imports")
def execute_directory_import(
    body: Dict[str, Any],
    actor: Actor = Depends(require_admin),
):
    estimate_id = str(body.get("estimateId") or "")
    kb_id = str(body.get("kbId") or "")
    selected = body.get("files")
    if not estimate_id or not kb_id or not isinstance(selected, list):
        raise ApiException(400, "INVALID_INPUT", "estimateId、kbId、files 必填")
    files = consume_import_preview(
        actor.tenant_id,
        actor.user_id,
        estimate_id,
        [str(value) for value in selected],
    )
    results = []
    for index, item in enumerate(files):
        results.append(
            KnowledgeService.upload_document(
                actor=actor,
                kb_id=kb_id,
                document_id=None,
                filename=item["filename"],
                file_bytes=item["bytes"],
                idempotency_key=f"{estimate_id}-{index}",
                build_config={},
            )
        )
    return results


@router.post("/admin/wiki-imports")
async def import_wiki_package(
    file: UploadFile = File(...),
    kbId: str = Form(...),
    buildConfig: str = Form(default="{}"),
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: Actor = Depends(require_admin),
):
    try:
        cfg = json.loads(buildConfig) if buildConfig else {}
    except Exception as exc:
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是合法 JSON") from exc
    if not isinstance(cfg, dict):
        raise ApiException(400, "INVALID_CONFIG", "buildConfig 必须是 JSON 对象")

    pages = split_wiki_package(await file.read())
    results = []
    for index, page in enumerate(pages):
        results.append(
            KnowledgeService.upload_document(
                actor=actor,
                kb_id=kbId,
                document_id=None,
                filename=page["filename"],
                file_bytes=page["bytes"],
                idempotency_key=f"{idempotency_key}-{index}",
                build_config=cfg,
            )
        )
    return results




@router.post("/internal/deletions")
def receive_internal_deletion(
    body: Dict[str, Any],
    request: Request,
):
    token = os.getenv("RAG_DELETION_API_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if (
        len(token) < 32
        or request.headers.get("Origin") is not None
        or not hmac.compare_digest(supplied, f"Bearer {token}")
    ):
        raise ApiException(403, "OPERATION_FORBIDDEN", "内部服务身份无效")
    return KnowledgeService.receive_deletion(
        str(body.get("originSystem") or ""),
        str(body.get("originTenantId") or ""),
        str(body.get("originDocumentId") or ""),
    )

# 后台任务轮询接口：前端通过 job 状态观察导入、构建和清理进度，不在 HTTP 请求里长时间阻塞。
@router.get("/admin/jobs")
def list_jobs(actor: Actor = Depends(require_admin)):
    return KnowledgeService.list_jobs(actor)


@router.get("/admin/jobs/{job_id}")
def get_job(job_id: str, actor: Actor = Depends(require_admin)):
    return KnowledgeService.get_job(actor, job_id)


@router.post("/admin/jobs/{job_id}/retry")
def retry_job(job_id: str, actor: Actor = Depends(require_admin)):
    return KnowledgeService.control_job(actor, job_id, cancel=False)


@router.post("/admin/jobs/{job_id}/cancel")
def cancel_job(job_id: str, actor: Actor = Depends(require_admin)):
    return KnowledgeService.control_job(actor, job_id, cancel=True)


# 文档预览接口：读取已持久化的解析/切分结果用于核对，不会重新触发 Docling。
@router.get("/documents/{doc_id}/versions/{version_id}/preview")
def get_preview(doc_id: str, version_id: str, actor: Actor = Depends(get_current_actor)):
    return KnowledgeService.get_preview(actor, doc_id, version_id)


@router.get("/documents/{doc_id}/artifacts/{artifact_id}")
def download_artifact(
    doc_id: str,
    artifact_id: str,
    actor: Actor = Depends(get_current_actor),
):
    data = KnowledgeService.get_artifact(actor, doc_id, artifact_id)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="source.bin"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@router.post("/admin/builds/{build_id}/accept-warnings")
def accept_build_warnings(
    build_id: str,
    body: Dict[str, Any],
    actor: Actor = Depends(require_admin),
):
    return KnowledgeService.accept_warnings(actor, build_id, body.get("reason", ""))


# 文档级 ACL 接口：在知识库 ACL 之上进一步收敛单文档访问范围，并通过版本字段避免并发覆盖。
@router.get("/admin/documents/{doc_id}/acl")
def get_document_acl(doc_id: str, actor: Actor = Depends(require_admin)):
    return KnowledgeService.get_document_acl(actor, doc_id)


@router.put("/admin/documents/{doc_id}/acl")
def update_document_acl(doc_id: str, payload: UpdateDocAclRequest, actor: Actor = Depends(require_admin)):
    return KnowledgeService.update_document_acl(actor, doc_id, payload.expectedVersion, payload.mode, payload.readers)


# 文档生命周期接口：trash/restore 属于可逆状态变化，purge 是不可逆物理清理，三者必须严格区分。
@router.post("/admin/documents/{doc_id}/{action}")
def document_lifecycle(doc_id: str, action: str, payload: LifecycleActionRequest, actor: Actor = Depends(require_admin)):
    return KnowledgeService.document_lifecycle(actor, doc_id, action, payload.expectedVersion)
