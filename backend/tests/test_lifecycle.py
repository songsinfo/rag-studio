import json
import time
import uuid
import os
from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from app.models import (
    tenant_table,
    app_user_table,
    knowledge_base_table,
    kb_grant_table,
    document_table,
    document_grant_table,
    index_build_table,
    chunk_table,
    source_unit_table,
    qa_run_table,
    run_source_table,
    run_event_table,
    feedback_table,
    audit_event_table,
    deletion_record_table,
    artifact_table,
)
import app.db as app_db
from app.knowledge.artifacts import save_artifact, read_artifact, delete_artifact
from app.retrieval.milvus_service import MilvusHybridService


def get_auth_context(client: TestClient, login_name: str = "admin", password: str = "admin123"):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": login_name, "password": password},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_resp.status_code == 200
    csrf_after = client.get("/api/v1/auth/csrf")
    return csrf_after.json()["csrfToken"]


def test_feedback_lifecycle(client):
    csrf = get_auth_context(client, "admin", "admin123")
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"

    kb_id = "kb-fb-1"
    doc_id = "doc-fb-1"
    build_id = "build-fb-1"

    with app_db.engine.begin() as conn:
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb_id, name="Feedback KB", status="LIVE", row_version=1, created_at=now, updated_at=now))
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb_id,
                principal_type="USER",
                user_id="user-admin-1",
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc_id,
                kb_id=kb_id,
                title="Feedback Doc",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=tenant_id,
                id=build_id,
                document_id=doc_id,
                version_id="ver-fb-1",
                build_no=1,
                config_id="cfg-fb-1",
                status="READY",
                capabilities={},
                fence=0,
                published_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            chunk_table.insert().values(
                tenant_id=tenant_id,
                id="chunk-fb-1",
                build_id=build_id,
                parent_id=None,
                kind="CHILD",
                ordinal=1,
                text_content="User feedback and evaluation loop ensures answer quality.",
                token_count=10,
                content_hash=b"hashfb",
                created_at=now,
                updated_at=now,
            )
        )

    MilvusHybridService.rebuild_from_mysql(tenant_id)

    # 第 1 步：创建会话并提交 run，准备后续反馈/权限/生命周期验证。
    conv_resp = client.post(
        "/api/v1/conversations",
        json={"title": "Feedback Conversation", "scope": {"mode": "ALL"}},
        headers={"X-CSRF-Token": csrf},
    )
    conv_id = conv_resp.json()["conversationId"]
    segment_id = conv_resp.json()["currentSegmentId"]

    run_resp = client.post(
        f"/api/v1/conversations/{conv_id}/runs",
        json={"question": "如何保证问答质量？", "segmentId": segment_id, "mode": "RAG"},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    run_id = run_resp.json()["runId"]

    # 轮询 run 到终态，确保断言针对完整执行结果而不是仅针对任务创建。
    for _ in range(60):
        time.sleep(0.1)
        r = client.get(f"/api/v1/runs/{run_id}")
        if r.json()["status"] in ("SUCCEEDED", "COMPLETED", "FAILED"):
            break

    run_detail = client.get(f"/api/v1/runs/{run_id}").json()
    assert run_detail["status"] in ("SUCCEEDED", "COMPLETED")
    assert run_detail["visibility"] == "VISIBLE"

    # 第 2 步：首次提交反馈，验证 run 与反馈记录的关联。
    fb_resp = client.put(
        f"/api/v1/runs/{run_id}/feedback",
        json={"rating": "USEFUL", "issueType": "", "comment": "回答很清晰"},
        headers={"X-CSRF-Token": csrf},
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "OPEN"

    # 第 3 步：再次提交同一 run 的反馈，验证 upsert 会更新而不是重复插入。
    fb_update_resp = client.put(
        f"/api/v1/runs/{run_id}/feedback",
        json={"rating": "USELESS", "issueType": "ANSWER_ERROR", "comment": "回答有误"},
        headers={"X-CSRF-Token": csrf},
    )
    assert fb_update_resp.status_code == 200

    # 第 4 步：管理员读取反馈列表与详情，验证运营侧可见性。
    admin_list = client.get("/api/v1/admin/feedback")
    assert admin_list.status_code == 200
    assert len(admin_list.json()) >= 1
    found_item = next(item for item in admin_list.json() if item.get("runId") == run_id)
    fb_id = found_item["id"]
    assert found_item["rating"] == "USELESS"
    assert found_item["issueType"] == "ANSWER_ERROR"

    admin_detail = client.get(f"/api/v1/admin/feedback/{fb_id}")
    assert admin_detail.status_code == 200
    assert admin_detail.json()["comment"] == "回答有误"

    # 第 5 步：管理员更新反馈处理状态，验证反馈闭环字段。
    patch_resp = client.patch(
        f"/api/v1/admin/feedback/{fb_id}",
        json={"status": "RESOLVED", "resolution": "已修复知识库数据"},
        headers={"X-CSRF-Token": csrf},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "RESOLVED"

    # 重新读取详情，确认状态更新已经持久化。
    admin_detail2 = client.get(f"/api/v1/admin/feedback/{fb_id}")
    assert admin_detail2.json()["status"] == "RESOLVED"
    assert admin_detail2.json()["resolution"] == "已修复知识库数据"


def test_acl_optimistic_locking_and_audit(client):
    csrf = get_auth_context(client, "admin", "admin123")
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"

    kb_id = "kb-acl-1"
    doc_id = "doc-acl-1"

    with app_db.engine.begin() as conn:
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb_id, name="ACL KB", status="LIVE", row_version=1, created_at=now, updated_at=now))
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc_id,
                kb_id=kb_id,
                title="ACL Document",
                acl_mode="RESTRICT",
                status="LIVE",
                active_build_id=None,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )

    # 第 1 步：用错误 expectedVersion 更新 KB ACL，预期发生乐观锁冲突。
    bad_kb_resp = client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": 99, "readers": [{"type": "USER", "userId": "user-admin-1"}]},
        headers={"X-CSRF-Token": csrf},
    )
    assert bad_kb_resp.status_code == 409
    assert bad_kb_resp.json()["code"] == "CONCURRENT_CONFLICT"

    # 第 2 步：使用正确版本更新 KB ACL，预期成功。
    kb_resp = client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": 1, "readers": [{"type": "USER", "userId": "user-admin-1"}]},
        headers={"X-CSRF-Token": csrf},
    )
    assert kb_resp.status_code == 200
    assert kb_resp.json()["rowVersion"] == 2

    # 第 3 步：文档 ACL 使用错误 expectedVersion，预期 409 冲突。
    bad_doc_resp = client.put(
        f"/api/v1/admin/documents/{doc_id}/acl",
        json={"expectedVersion": 99, "mode": "RESTRICT", "readers": [{"type": "USER", "userId": "user-admin-1"}]},
        headers={"X-CSRF-Token": csrf},
    )
    assert bad_doc_resp.status_code == 409

    # 第 4 步：文档 ACL 使用正确版本更新，预期成功。
    doc_resp = client.put(
        f"/api/v1/admin/documents/{doc_id}/acl",
        json={"expectedVersion": 1, "mode": "RESTRICT", "readers": [{"type": "USER", "userId": "user-admin-1"}]},
        headers={"X-CSRF-Token": csrf},
    )
    assert doc_resp.status_code == 200
    assert doc_resp.json()["rowVersion"] == 2
    assert doc_resp.json()["mode"] == "RESTRICT"

    # 第 5 步：检查安全相关 ACL 变更都生成审计事件。
    with app_db.engine.begin() as conn:
        audits = conn.execute(
            audit_event_table.select().where(audit_event_table.c.tenant_id == tenant_id)
        ).fetchall()
        actions = [a.action for a in audits]
        assert "ACL_UPDATE" in actions


def test_revocation_desensitization(client):
    csrf = get_auth_context(client, "admin", "admin123")
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"

    kb_id = "kb-revoke-1"
    doc_id = "doc-revoke-1"
    build_id = "build-revoke-1"

    with app_db.engine.begin() as conn:
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb_id, name="Revoke KB", status="LIVE", row_version=1, created_at=now, updated_at=now))
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb_id,
                principal_type="USER",
                user_id="user-admin-1",
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc_id,
                kb_id=kb_id,
                title="Confidential Product Roadmap",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=tenant_id,
                id=build_id,
                document_id=doc_id,
                version_id="ver-revoke-1",
                build_no=1,
                config_id="cfg-revoke-1",
                status="READY",
                capabilities={},
                fence=0,
                published_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            chunk_table.insert().values(
                tenant_id=tenant_id,
                id="chunk-revoke-1",
                build_id=build_id,
                parent_id=None,
                kind="CHILD",
                ordinal=1,
                text_content="Confidential project details: Secret Project Apollo launches in Q4.",
                token_count=12,
                content_hash=b"hashrevoke",
                created_at=now,
                updated_at=now,
            )
        )

    MilvusHybridService.rebuild_from_mysql(tenant_id)

    # 第 1 步：创建会话并提交 run，准备后续反馈/权限/生命周期验证。
    conv_resp = client.post(
        "/api/v1/conversations",
        json={"title": "Project Apollo Query", "scope": {"mode": "ALL"}},
        headers={"X-CSRF-Token": csrf},
    )
    conv_id = conv_resp.json()["conversationId"]
    segment_id = conv_resp.json()["currentSegmentId"]

    run_resp = client.post(
        f"/api/v1/conversations/{conv_id}/runs",
        json={"question": "Apollo 项目什么时候发布？", "segmentId": segment_id, "mode": "RAG"},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": str(uuid.uuid4())},
    )
    run_id = run_resp.json()["runId"]

    for _ in range(60):
        time.sleep(0.1)
        r = client.get(f"/api/v1/runs/{run_id}")
        if r.json()["status"] in ("SUCCEEDED", "COMPLETED", "FAILED"):
            break

    run_data = client.get(f"/api/v1/runs/{run_id}").json()
    assert run_data["status"] in ("SUCCEEDED", "COMPLETED")
    assert run_data["visibility"] == "VISIBLE"
    assert "Apollo" in run_data["question"]
    assert "answer" in run_data

    # 第 2 步：把 KB 授权改给无关用户，模拟历史会话引用的数据被撤权。
    client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": 1, "readers": [{"type": "USER", "userId": "user-disabled-1"}]},
        headers={"X-CSRF-Token": csrf},
    )

    # 第 3 步：确认撤权后 run 被脱敏为 HIDDEN，并且 question/answer 正文都不再返回。
    run_after_revoke = client.get(f"/api/v1/runs/{run_id}").json()
    assert run_after_revoke["visibility"] == "HIDDEN"
    assert "question" not in run_after_revoke
    assert "answer" not in run_after_revoke
    assert "notice" in run_after_revoke

    # 第 4 步：确认会话标题也被遮蔽，避免通过标题泄露原问题内容。
    conv_after = client.get(f"/api/v1/conversations/{conv_id}").json()
    assert conv_after["title"] is None

    conv_list = client.get("/api/v1/conversations").json()
    target_conv = next(c for c in conv_list if c["id"] == conv_id)
    assert target_conv["title"] is None

    # 第 5 步：证据详情接口对已撤权内容返回 404，避免暴露资源是否存在。
    ev_resp = client.get(f"/api/v1/runs/{run_id}/evidence/E1")
    assert ev_resp.status_code == 404

    # 第 6 步：对隐藏 run 提交反馈也返回 404，不能借反馈接口确认历史内容。
    fb_fail = client.put(
        f"/api/v1/runs/{run_id}/feedback",
        json={"rating": "USEFUL"},
        headers={"X-CSRF-Token": csrf},
    )
    assert fb_fail.status_code == 404


def test_document_lifecycle_trash_restore_purge(client):
    csrf = get_auth_context(client, "admin", "admin123")
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"

    kb_id = "kb-life-1"
    doc_id = "doc-life-1"
    build_id = "build-life-1"

    # 写入真实 artifact 文件，后续 purge 测试同时验证数据库与物理文件清理。
    dummy_bytes = b"Artifact content for lifecycle verification test."
    storage_key, _, _ = save_artifact(tenant_id, "test_doc.txt", dummy_bytes)

    with app_db.engine.begin() as conn:
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb_id, name="Lifecycle KB", status="LIVE", row_version=1, created_at=now, updated_at=now))
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb_id,
                principal_type="USER",
                user_id="user-admin-1",
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc_id,
                kb_id=kb_id,
                title="Lifecycle Target Document",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            artifact_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                document_id=doc_id,
                kind="SOURCE",
                storage_key=storage_key,
                sha256=b"art_hash",
                size_bytes=len(dummy_bytes),
                status="READY",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=tenant_id,
                id=build_id,
                document_id=doc_id,
                version_id="ver-life-1",
                build_no=1,
                config_id="cfg-life-1",
                status="READY",
                capabilities={},
                fence=0,
                published_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            chunk_table.insert().values(
                tenant_id=tenant_id,
                id="chunk-life-1",
                build_id=build_id,
                parent_id=None,
                kind="CHILD",
                ordinal=1,
                text_content="Lifecycle testing: trash, restore, and local purge mechanics.",
                token_count=10,
                content_hash=b"hashlife",
                created_at=now,
                updated_at=now,
            )
        )

    MilvusHybridService.rebuild_from_mysql(tenant_id)

    # 第 1 步：文档处于 ACTIVE 时检索应正常命中。
    s_resp = client.get("/api/v1/search?query=Lifecycle")
    assert any(e["chunkId"] == "chunk-life-1" for e in s_resp.json()["evidence"])

    # 第 2 步：把文档移入回收站，验证软删除状态变化。
    trash_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/trash",
        json={"expectedVersion": 1},
        headers={"X-CSRF-Token": csrf},
    )
    assert trash_resp.status_code == 200
    assert trash_resp.json()["status"] == "TRASHED"

    # 文档进入 TRASHED 后检索必须立即不可见，不能等待物理 purge 才隐藏。
    s_trashed = client.get("/api/v1/search?query=Lifecycle")
    assert not any(e["chunkId"] == "chunk-life-1" for e in s_trashed.json()["evidence"])

    # 第 3 步：恢复文档，验证 active build 和检索可见性可重新启用。
    # 注意：第一次 trash 已把 document.row_version 提升到 2，restore 必须基于最新版本。
    restore_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/restore",
        json={"expectedVersion": 2},
        headers={"X-CSRF-Token": csrf},
    )
    assert restore_resp.status_code == 200
    assert restore_resp.json()["status"] == "LIVE"

    # restore 完成后检索应再次命中，证明恢复不仅改变状态，也恢复有效可见性。
    s_restored = client.get("/api/v1/search?query=Lifecycle")
    assert any(e["chunkId"] == "chunk-life-1" for e in s_restored.json()["evidence"])

    # 第 4 步：purge 前再次 trash；永久删除只允许从 TRASHED 状态进入。
    # restore 后 row_version 已变为 3，第二次 trash 需要使用该最新版本。
    client.post(
        f"/api/v1/admin/documents/{doc_id}/trash",
        json={"expectedVersion": 3},
        headers={"X-CSRF-Token": csrf},
    )

    # 第 5 步：执行不可逆 purge，验证完整隐私清理链路。
    # 第二次 trash 后 row_version 变为 4，purge 必须基于该版本防止并发误删。
    purge_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/purge",
        json={"expectedVersion": 4},
        headers={"X-CSRF-Token": csrf},
    )
    assert purge_resp.status_code == 200
    assert purge_resp.json()["status"] == "PURGED"

    # 确认对象存储/文件系统中的物理 artifact 已删除，不能只清元数据。
    with pytest.raises(Exception):
        read_artifact(storage_key)

    # 确认 MySQL 中正文、chunk、版本等派生数据均已清除或脱敏。
    with app_db.engine.begin() as conn:
        doc_row = conn.execute(document_table.select().where(document_table.c.id == doc_id)).first()
        assert doc_row.status == "PURGED"
        assert doc_row.title is None
        assert doc_row.active_build_id is None

        chunks = conn.execute(chunk_table.select().where(chunk_table.c.build_id == build_id)).fetchall()
        for c in chunks:
            assert c.text_content == ""

        del_rec = conn.execute(deletion_record_table.select().where(deletion_record_table.c.document_id == doc_id)).first()
        assert del_rec is not None
        assert del_rec.status == "COMPLETE"
