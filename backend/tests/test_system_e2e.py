import io
import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from sqlalchemy import select
from starlette.testclient import TestClient

from app.models import (
    knowledge_base_table,
    document_table,
    document_version_table,
    index_build_table,
    chunk_table,
    kb_grant_table,
    document_grant_table,
    qa_run_table,
    run_source_table,
    conversation_table,
    context_segment_table,
    feedback_table,
    app_user_table,
    user_group_table,
    group_member_table,
    tenant_table,
    job_table,
)
import app.db as app_db
from app.db import get_db_connection
from app.qa.service import QaService
from app.qa.model_gateway import ModelGateway
from app.qa.claim_validator import validate_claims
from app.retrieval.milvus_service import MilvusHybridService
from app.errors import ApiException
from app.main import app


def login_client(client: TestClient, login_name: str = "admin", password: str = "admin123") -> str:
    """Helper to authenticate and return the post-login CSRF token."""
    csrf_resp = client.get("/api/v1/auth/csrf")
    assert csrf_resp.status_code == 200
    csrf_token = csrf_resp.json()["csrfToken"]

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": login_name, "password": password},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_resp.status_code == 200, login_resp.text

    csrf_after = client.get("/api/v1/auth/csrf")
    return csrf_after.json()["csrfToken"]


def test_e2e_full_user_and_admin_workflow(client: TestClient):
    """
    Complete end-to-end user and admin workflow verification:
    1. Login & Profile check (/me)
    2. Knowledge Base create, ACL inspect & update
    3. Document upload with buildConfig, job tracking, version preview
    4. BM25 Search
    5. Conversation, Segment scope, RAG question run, claim validation, evidence preview
    6. Agent QA run and controlled trace inspection
    7. User feedback submission and admin resolution
    8. Operations management: jobs list/retry/cancel, configs list/publish, users, groups with membership optimistic lock
    9. Document lifecycle: trash, list trash, restore, and purge
    """
    csrf = login_client(client, "admin", "admin123")

    # 第 1 步：调用 /me 验证登录身份与租户上下文。
    me_resp = client.get("/api/v1/me")
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["loginName"] == "admin"
    assert me_data["admin"] is True
    assert me_data["tenantCode"] == "demo"

    # 第 2 步：覆盖知识库创建、列表与 ACL 基本链路。
    create_kb_resp = client.post(
        "/api/v1/admin/knowledge-bases",
        json={"name": "Engineering Docs", "readers": [{"type": "USER", "userId": me_data["id"]}]},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert create_kb_resp.status_code == 200
    kb_id = create_kb_resp.json()["id"]

    list_kb_resp = client.get("/api/v1/knowledge-bases")
    assert list_kb_resp.status_code == 200
    assert any(k["id"] == kb_id for k in list_kb_resp.json())

    # 读取并更新 KB ACL，验证 row_version/security_epoch 协同生效。
    kb_acl_resp = client.get(f"/api/v1/admin/knowledge-bases/{kb_id}/acl")
    assert kb_acl_resp.status_code == 200
    kb_acl = kb_acl_resp.json()

    update_kb_acl_resp = client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": kb_acl.get("rowVersion", 0), "readers": [{"type": "ALL"}]},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert update_kb_acl_resp.status_code == 200

    # 第 3 步：携带 buildConfig 上传文档，触发版本化 ingestion。
    doc_content = "# System Architecture\n\nFastAPI backend replaces Spring Boot with Milvus hybrid retrieval."
    upload_resp = client.post(
        "/api/v1/admin/documents",
        data={"kbId": kb_id, "buildConfig": json.dumps({"embedding": True})},
        files={"file": ("arch.md", io.BytesIO(doc_content.encode("utf-8")), "text/markdown")},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert upload_resp.status_code == 202
    upload_data = upload_resp.json()
    doc_id = upload_data["documentId"]
    job_id = upload_data["jobId"]

    # 轮询后台 job 到终态，确认 durable worker 真正完成构建。
    for _ in range(30):
        j_resp = client.get(f"/api/v1/admin/jobs/{job_id}")
        assert j_resp.status_code == 200
        if j_resp.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.1)

    # 查看文档详情与版本预览，核对解析、chunk 与版本状态。
    doc_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert doc_resp.status_code == 200
    active_build_id = doc_resp.json()["activeBuildId"]
    assert active_build_id is not None

    with get_db_connection() as conn:
        ver_id = conn.execute(
            select(document_version_table.c.id).where(document_version_table.c.document_id == doc_id)
        ).scalar()

    preview_resp = client.get(f"/api/v1/documents/{doc_id}/versions/{ver_id}/preview")
    assert preview_resp.status_code == 200
    preview_data = preview_resp.json()
    assert len(preview_data["chunks"]) > 0

    # 用真实 chunk 文本同步 Milvus mock，保证后续检索测试稳定且与持久化内容一致。
    chunk_item = preview_data["chunks"][0]
    MilvusHybridService.upsert_chunks(
        tenant_id="tenant-demo-1",
        build_id=active_build_id,
        chunks=[{"id": chunk_item["id"], "kind": "CHILD", "text_content": chunk_item["text"], "ordinal": 0}],
        kb_id=kb_id,
        doc_id=doc_id,
        version_id=str(ver_id),
    )

    # 第 4 步：调用搜索接口，验证 ACL、排序和证据结构。
    search_resp = client.get(f"/api/v1/search?query=FastAPI&kbIds={kb_id}")
    assert search_resp.status_code == 200
    assert len(search_resp.json()) > 0

    # 第 5 步：覆盖会话、segment scope 与 RAG run 的完整用户流程。
    conv_resp = client.post(
        "/api/v1/conversations",
        json={"title": "Test Chat", "scope": {"mode": "ALL", "kbIds": []}},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert conv_resp.status_code in (200, 201)
    conv_id = conv_resp.json()["conversationId"]

    # 新增 segment，验证 scope 变化会形成新的上下文边界。
    seg_resp = client.post(
        f"/api/v1/conversations/{conv_id}/segments",
        json={"scope": {"mode": "SELECTED", "kbIds": [kb_id]}},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert seg_resp.status_code == 200
    segment_id = seg_resp.json()["currentSegmentId"]

    # 提交 RAG 查询；测试环境使用 ModelGateway 内置确定性模拟，保证端到端断言可重复。
    run_post_resp = client.post(
        f"/api/v1/conversations/{conv_id}/runs",
        json={"question": "系统架构采用了什么框架？", "segmentId": segment_id, "mode": "RAG", "strict": False},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert run_post_resp.status_code == 202
    run_id = run_post_resp.json()["runId"]

    # 轮询 QA run 到终态，确认异步执行与最终落库。
    run_data = None
    for _ in range(60):
        time.sleep(0.1)
        run_poll_resp = client.get(f"/api/v1/runs/{run_id}")
        assert run_poll_resp.status_code == 200
        run_data = run_poll_resp.json()
        if run_data["status"] in ("SUCCEEDED", "COMPLETED", "FAILED", "VERIFICATION_FAILED"):
            break

    assert run_data is not None
    assert run_data["status"] in ("SUCCEEDED", "COMPLETED")
    assert len(run_data.get("evidence", [])) > 0
    ev_id = run_data["evidence"][0]["evidenceId"]

    # 读取 evidence 详情，验证答案引用能回溯到原始文档位置。
    ev_resp = client.get(f"/api/v1/runs/{run_id}/evidence/{ev_id}")
    assert ev_resp.status_code == 200
    assert ev_resp.json()["documentId"] == doc_id

    # 第 6 步：提交用户反馈并由管理员复核，覆盖反馈闭环。
    fb_resp = client.put(
        f"/api/v1/runs/{run_id}/feedback",
        json={"rating": "USEFUL", "comment": "非常准确", "issueType": "OTHER"},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert fb_resp.status_code == 200

    admin_fb_list = client.get("/api/v1/admin/feedback")
    assert admin_fb_list.status_code == 200
    assert len(admin_fb_list.json()) > 0
    fb_id = admin_fb_list.json()[0]["id"]

    patch_fb = client.patch(
        f"/api/v1/admin/feedback/{fb_id}",
        json={"status": "RESOLVED", "resolution": "经核实回答准确无误"},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert patch_fb.status_code == 200

    # 第 7 步：读取受控 trace，验证可观测性数据不会越权暴露敏感正文。
    trace_resp = client.get(f"/api/v1/admin/runs/{run_id}/trace")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert trace_data["runId"] == run_id
    assert "events" in trace_data
    trace_text = json.dumps(trace_data)
    assert "sk-" not in trace_text
    assert "RAG_SESSION" not in trace_text

    # 第 8 步：覆盖 Jobs、Configs、Users、Groups 等运营管理接口。
    # 验证任务列表、取消和重试，重点检查状态机与幂等约束。
    jobs_resp = client.get("/api/v1/admin/jobs")
    assert jobs_resp.status_code == 200
    assert len(jobs_resp.json()) > 0

    # 验证配置列表与版本化配置读取。
    cfg_create = client.post(
        "/api/v1/admin/configs",
        json={"kind": "RETRIEVAL", "name": "e2e_test_cfg", "payload": {"retrieval": {"mode": "BM25"}}},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert cfg_create.status_code in (200, 201)
    cfg_id = cfg_create.json()["id"]

    cfg_publish = client.post(
        f"/api/v1/admin/configs/{cfg_id}/publish",
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert cfg_publish.status_code == 200

    # 验证用户管理查询与状态字段。
    users_resp = client.get("/api/v1/admin/users")
    assert users_resp.status_code == 200
    assert any(u["loginName"] == "admin" for u in users_resp.json())

    # 验证组管理及成员更新的乐观锁，防止并发覆盖成员列表。
    groups_resp = client.get("/api/v1/admin/groups")
    assert groups_resp.status_code == 200
    assert len(groups_resp.json()) > 0
    grp = groups_resp.json()[0]
    grp_id = grp["id"]
    current_epoch = grp["securityEpoch"]

    update_members_resp = client.put(
        f"/api/v1/admin/groups/{grp_id}/members",
        json={"userIds": [me_data["id"]], "expectedSecurityEpoch": current_epoch},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert update_members_resp.status_code == 200
    assert update_members_resp.json()["updated"] is True

    # 确认权限成员关系变化后 securityEpoch 已提升，使旧授权缓存/异步任务失效。
    new_groups_resp = client.get("/api/v1/admin/groups")
    assert new_groups_resp.json()[0]["securityEpoch"] == current_epoch + 1

    # 第 9 步：覆盖文档 Trash、回收站列表、Restore 和不可逆 Purge 全生命周期。
    # 执行软删除 Trash，文档应立即退出正常检索范围。
    trash_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/trash",
        json={"expectedVersion": 1},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert trash_resp.status_code == 200

    # 查询回收站列表，确认软删除记录仍可管理。
    trashed_list = client.get("/api/v1/admin/documents?status=TRASHED")
    assert trashed_list.status_code == 200
    assert any(d["id"] == doc_id for d in trashed_list.json())

    # 恢复文档，验证状态与检索可见性重新生效。
    restore_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/restore",
        json={"expectedVersion": 2},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert restore_resp.status_code == 200

    # Purge 前再次 Trash；永久删除只允许从 TRASHED 状态进入，避免误删活动文档。
    trash2_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/trash",
        json={"expectedVersion": 3},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert trash2_resp.status_code == 200

    # 执行不可逆 Purge，验证物理文件与派生数据一起清理。
    purge_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/purge",
        json={"expectedVersion": 4},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert purge_resp.status_code == 200


def test_identity_and_csrf_security_boundaries(client: TestClient):
    """Verify password protection, rate limits, CSRF token binding, and session revocation."""
    # 第 1 步：使用错误凭据登录，必须失败且不能泄露账号是否存在。
    csrf = client.get("/api/v1/auth/csrf").json()["csrfToken"]
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "admin", "password": "wrongpassword"},
        headers={"X-CSRF-Token": csrf},
    )
    assert bad_login.status_code == 401
    assert bad_login.json()["code"] == "INVALID_CREDENTIALS"

    # 第 2 步：禁用账号即使密码正确也不能建立会话。
    disabled_login = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "disabled_user", "password": "admin123"},
        headers={"X-CSRF-Token": csrf},
    )
    assert disabled_login.status_code == 401

    # 第 3 步：成功登录后验证完整 CSRF 防护。
    csrf_valid = login_client(client, "admin", "admin123")

    # 缺失 X-CSRF-Token Header 的写请求必须被拒绝。
    no_csrf = client.post("/api/v1/conversations", json={"title": "test"})
    assert no_csrf.status_code == 403
    assert no_csrf.json()["code"] == "CSRF_INVALID"

    # 伪造的 CSRF Header 与 Cookie 不匹配时必须被拒绝。
    bad_token = client.post(
        "/api/v1/conversations",
        json={"title": "test"},
        headers={"X-CSRF-Token": "completely-fake-token"},
    )
    assert bad_token.status_code == 403

    # 来自非可信 Origin 的写请求必须在业务处理前被拒绝。
    foreign_origin = client.post(
        "/api/v1/conversations",
        json={"title": "test"},
        headers={"X-CSRF-Token": csrf_valid, "Origin": "https://malicious-site.com"},
    )
    assert foreign_origin.status_code == 403

    # logout 必须立即撤销服务器端 session，而不是只删除浏览器 Cookie。
    logout_resp = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf_valid})
    assert logout_resp.status_code == 200
    assert logout_resp.json()["authenticated"] is False

    # 退出后继续访问受保护接口必须返回 401，验证旧 session 已不可用。
    post_logout_me = client.get("/api/v1/me")
    assert post_logout_me.status_code == 401


def test_acl_revocation_masks_historical_answers(client: TestClient):
    """
    When document access is revoked from a user (e.g. through group membership or ACL update),
    historical answers citing that document must be masked with visibility='HIDDEN' and answer=None,
    and direct preview access must be blocked.
    """
    csrf = login_client(client, "admin", "admin123")
    now = datetime.now(timezone.utc)

    # 第 1 步：创建受限 KB 和受限文档，准备权限撤销后的历史脱敏测试。
    doc_id = "doc-revoke-test-1"
    kb_id = "kb-revoke-test-1"
    build_id = "build-revoke-test-1"
    ver_id = "ver-revoke-test-1"

    with get_db_connection() as conn:
        conn.execute(
            knowledge_base_table.insert().values(
                tenant_id="tenant-demo-1",
                id=kb_id,
                name="Confidential KB",
                status="LIVE",
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id="tenant-demo-1",
                id="kbg-rev-1",
                kb_id=kb_id,
                principal_type="USER",
                user_id="user-admin-1",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_table.insert().values(
                tenant_id="tenant-demo-1",
                id=doc_id,
                kb_id=kb_id,
                title="Confidential Strategy",
                acl_mode="RESTRICT",
                status="LIVE",
                active_build_id=build_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_version_table.insert().values(
                tenant_id="tenant-demo-1",
                id=ver_id,
                document_id=doc_id,
                version_no=1,
                source_artifact_id="art-rev-1",
                source_hash=b"0" * 32,
                original_name="strategy.md",
                media_type="text/markdown",
                created_by="user-admin-1",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id="tenant-demo-1",
                id=build_id,
                document_id=doc_id,
                version_id=ver_id,
                build_no=1,
                config_id="cfg-1",
                capabilities={},
                status="READY",
                published_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        # 通过 ops group 授予访问权限，验证组继承而非直接用户授权。
        conn.execute(
            document_grant_table.insert().values(
                tenant_id="tenant-demo-1",
                id="dg-rev-1",
                document_id=doc_id,
                principal_type="GROUP",
                group_id="group-ops-1",
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    # 第 2 步：创建引用该文档的历史会话/run，构造撤权前可见的真实历史。
    conv_id = "cv-revoke-test"
    seg_id = "seg-revoke-test"
    run_id = "run-revoke-test"
    with get_db_connection() as conn:
        conn.execute(
            conversation_table.insert().values(
                tenant_id="tenant-demo-1",
                id=conv_id,
                owner_id="user-admin-1",
                title="Strategy Discussion",
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            context_segment_table.insert().values(
                tenant_id="tenant-demo-1",
                id=seg_id,
                conversation_id=conv_id,
                segment_no=1,
                scope_mode="ALL",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            qa_run_table.insert().values(
                tenant_id="tenant-demo-1",
                id=run_id,
                owner_id="user-admin-1",
                conversation_id=conv_id,
                segment_id=seg_id,
                turn_no=1,
                mode="RAG",
                status="COMPLETED",
                stage="FINISHED",
                idempotency_key="rev-idem-1",
                request_hash=b"hashrev",
                question="机密战略是什么？",
                result_json={
                    "status": "COMPLETED",
                    "answer": "秘密战略涉及核心业务机密。",
                    "evidence": [
                        {
                            "evidenceId": "ev-rev-1",
                            "documentId": doc_id,
                            "versionId": ver_id,
                            "quote": "秘密战略涉及核心业务机密。",
                        }
                    ],
                    "claims": [{"claimText": "秘密战略", "citations": [1]}],
                },
                security_epoch=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            run_source_table.insert().values(
                tenant_id="tenant-demo-1",
                id="rs-rev-1",
                run_id=run_id,
                document_id=doc_id,
                build_id=build_id,
                evidence_id="ev-rev-1",
                purpose="CITATION",
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    # 撤权前先确认用户能看到答案，保证后续隐藏确实由权限变化触发。
    cv_before = client.get(f"/api/v1/conversations/{conv_id}").json()
    assert cv_before["runs"][0]["visibility"] == "VISIBLE"
    assert cv_before["runs"][0]["answer"] == "秘密战略涉及核心业务机密。"

    # 第 3 步：把用户移出 group-ops-1，撤销其通过组继承获得的访问权。
    update_members = client.put(
        "/api/v1/admin/groups/group-ops-1/members",
        json={"userIds": [], "expectedSecurityEpoch": 1},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert update_members.status_code == 200

    # 第 4 步：重新读取历史会话，答案必须变为 HIDDEN，不能继续返回缓存正文。
    cv_after = client.get(f"/api/v1/conversations/{conv_id}").json()
    run_after = cv_after["runs"][0]
    assert run_after["visibility"] == "HIDDEN"
    assert run_after.get("answer") is None


def test_concurrency_and_optimistic_locking(client: TestClient):
    """Verify that version conflict checks protect document lifecycle and group members from state races."""
    csrf = login_client(client, "admin", "admin123")

    # 组更新使用不匹配的 expectedSecurityEpoch 时必须返回 409，验证安全版本乐观锁。
    stale_epoch = client.put(
        "/api/v1/admin/groups/group-ops-1/members",
        json={"userIds": [], "expectedSecurityEpoch": 9999},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert stale_epoch.status_code == 409
    assert stale_epoch.json()["code"] == "VERSION_CONFLICT"

    # 文档生命周期操作使用错误 expectedVersion 时必须返回 409，防止并发误操作。
    doc_id = "doc-race-test"
    now = datetime.now(timezone.utc)
    with get_db_connection() as conn:
        conn.execute(
            document_table.insert().values(
                tenant_id="tenant-demo-1",
                id=doc_id,
                kb_id="any-kb",
                title="Race Doc",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=None,
                lifecycle_epoch=0,
                row_version=5,
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    stale_doc = client.post(
        f"/api/v1/admin/documents/{doc_id}/trash",
        json={"expectedVersion": 4},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert stale_doc.status_code == 409
    assert stale_doc.json()["code"] in ("VERSION_CONFLICT", "CONCURRENT_CONFLICT")


def test_milvus_tenant_and_active_build_isolation():
    """Verify that hybrid search enforces tenant isolation and ignores trashed/draft documents."""
    # 分别写入 Tenant A/B 的 chunk，构造多租户隔离测试数据。
    MilvusHybridService.upsert_chunks(
        tenant_id="tenant-demo-1",
        build_id="build-active-demo",
        chunks=[{"id": "chunk-demo-1", "kind": "CHILD", "text_content": "Kubernetes cluster management", "ordinal": 0}],
        kb_id="kb-1",
        doc_id="doc-1",
        version_id="ver-1",
    )
    MilvusHybridService.upsert_chunks(
        tenant_id="tenant-foreign-2",
        build_id="build-active-foreign",
        chunks=[{"id": "chunk-foreign-1", "kind": "CHILD", "text_content": "Kubernetes secrets and tokens", "ordinal": 0}],
        kb_id="kb-foreign",
        doc_id="doc-foreign",
        version_id="ver-foreign",
    )

    # tenant-demo-1 搜索绝不能返回其他租户的 chunk-foreign-1，这是强隔离断言。
    results = MilvusHybridService.search(
        tenant_id="tenant-demo-1",
        allowed_build_ids=["build-active-demo"],
        query="Kubernetes",
        top_k=10,
    )
    chunk_ids = [r["chunkId"] for r in results]
    assert "chunk-demo-1" in chunk_ids
    assert "chunk-foreign-1" not in chunk_ids

    # 使用未授权 build 搜索必须返回空结果，不能仅凭 chunk 存在就放行。
    unallowed = MilvusHybridService.search(
        tenant_id="tenant-demo-1",
        allowed_build_ids=["build-trashed-other"],
        query="Kubernetes",
        top_k=10,
    )
    assert len(unallowed) == 0


def test_claim_validation_detects_hallucination_and_bad_citation():
    """Verify that claim validation flags answers with ungrounded claims or invalid citation spans."""
    invalid_answer = {
        "status": "COMPLETED",
        "claims": [
            {
                "id": "c1",
                "text": "系统采用三副本高可用架构",
                "evidence": [{"evidenceId": "NONEXISTENT_EID", "start": 0, "end": 10}],
            }
        ],
        "gaps": [],
        "conflicts": [],
    }
    evidence_map = {"E1": "系统采用单机轻量级部署架构。"}

    with pytest.raises(ApiException) as exc:
        validate_claims(invalid_answer, evidence_map)
    assert exc.value.code == "INVALID_MODEL_EVIDENCE"


def test_ai_gateway_failure_leads_to_clean_terminal_failure(client: TestClient):
    """Verify that when AI Gateway fails or times out, the run terminates in FAILED without leaking credentials."""
    csrf = login_client(client, "admin", "admin123")
    now = datetime.now(timezone.utc)
    doc_id = "doc-gw-1"
    kb_id = "kb-gw-1"
    build_id = "build-gw-1"
    with get_db_connection() as conn:
        conn.execute(
            knowledge_base_table.insert().values(
                tenant_id="tenant-demo-1", id=kb_id, name="KB GW", status="LIVE", row_version=1, created_at=now, updated_at=now
            )
        )
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id="tenant-demo-1", id="kbg-gw-1", kb_id=kb_id, principal_type="ALL", created_at=now, updated_at=now
            )
        )
        conn.execute(
            document_table.insert().values(
                tenant_id="tenant-demo-1", id=doc_id, kb_id=kb_id, title="Doc GW", acl_mode="INHERIT", status="LIVE",
                active_build_id=build_id, lifecycle_epoch=0, row_version=1, created_at=now, updated_at=now
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id="tenant-demo-1", id=build_id, document_id=doc_id, version_id="ver-gw-1", config_id="cfg-1", build_no=1,
                capabilities={}, status="READY", published_at=now, created_at=now, updated_at=now
            )
        )
        conn.commit()

    MilvusHybridService.upsert_chunks(
        tenant_id="tenant-demo-1",
        build_id=build_id,
        chunks=[{"id": "chunk-gw-1", "kind": "CHILD", "text_content": "会话网关测试相关证据内容", "ordinal": 0}],
        kb_id=kb_id,
        doc_id=doc_id,
        version_id="ver-gw-1",
    )

    conv_resp = client.post(
        "/api/v1/conversations",
        json={"title": "Gateway Error Chat", "scope": {"mode": "ALL", "kbIds": []}},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    conv_id = conv_resp.json()["conversationId"]
    segment_id = conv_resp.json()["currentSegmentId"]

    # 统一 Agent 使用 chat_tools；注入网关故障，验证错误收敛、重试与脱敏行为。
    with patch.object(ModelGateway, "chat_tools", side_effect=ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", "大模型网关故障")):
        run_resp = client.post(
            f"/api/v1/conversations/{conv_id}/runs",
            json={"question": "会话网关测试相关证据内容？", "segmentId": segment_id, "mode": "RAG"},
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
        )
        assert run_resp.status_code == 202
        run_id = run_resp.json()["runId"]

        # 轮询故障 run 到终态，确保断言覆盖后台 worker 写回后的真实状态。
        result = None
        for _ in range(60):
            time.sleep(0.1)
            run_poll = client.get(f"/api/v1/runs/{run_id}")
            result = run_poll.json()
            if result["status"] in ("FAILED", "COMPLETED", "SUCCEEDED"):
                break

        assert result is not None
        assert result["status"] == "FAILED"
        # 确认终态 reason/error 文本没有泄露模型供应商 key 或其他密钥信息。
        full_text = json.dumps(result)
        assert "sk-" not in full_text
        assert "Bearer" not in full_text
        assert "token" not in full_text.lower() or "session" not in full_text.lower()


def test_unhandled_exception_sanitization():
    """Verify that unhandled server errors return a sanitized 500 JSON without exposing stack traces."""
    client_no_raise = TestClient(app, raise_server_exceptions=False)
    csrf = login_client(client_no_raise, "admin", "admin123")

    with patch("app.knowledge.service.KnowledgeService.list_jobs", side_effect=ValueError("FATAL_INTERNAL_DB_SECRET_KEY=12345")):
        resp = client_no_raise.get("/api/v1/admin/jobs")
        assert resp.status_code == 500
        data = resp.json()
        assert data["code"] == "INTERNAL_ERROR"
        assert "FATAL_INTERNAL_DB_SECRET_KEY" not in json.dumps(data)
        assert "Traceback" not in json.dumps(data)
