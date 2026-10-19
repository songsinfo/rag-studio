import time
import pytest
from app.identity.security import generate_token, hash_token


def get_auth_context(client, login_name="admin", password="admin123"):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": login_name, "password": password},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_resp.status_code == 200

    # 获取绑定到当前会话的新 CSRF token，避免复用登录前的匿名 token。
    fresh_csrf = client.get("/api/v1/auth/csrf").json()["csrfToken"]
    return fresh_csrf


def test_kb_creation_and_concurrency(client):
    csrf = get_auth_context(client, "admin", "admin123")

    # 第 1 步：创建知识库，作为文档导入与 ACL 测试的父资源。
    create_resp = client.post(
        "/api/v1/admin/knowledge-bases",
        json={"name": "test-kb-1", "readers": []},
        headers={"X-CSRF-Token": csrf},
    )
    assert create_resp.status_code == 200
    kb_data = create_resp.json()
    kb_id = kb_data["id"]
    assert kb_data["rowVersion"] == 1

    # 第 2 步：读取知识库 ACL 和当前版本号，为后续乐观锁更新做准备。
    acl_resp = client.get(f"/api/v1/admin/knowledge-bases/{kb_id}/acl")
    assert acl_resp.status_code == 200
    assert acl_resp.json()["rowVersion"] == 1

    # 第 3 步：使用错误 expectedVersion 更新 ACL，预期返回 409，验证并发覆盖保护。
    bad_update = client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": 999, "readers": []},
        headers={"X-CSRF-Token": csrf},
    )
    assert bad_update.status_code == 409
    assert bad_update.json()["code"] == "CONCURRENT_CONFLICT"

    # 第 4 步：使用正确 expectedVersion 更新 ACL，预期成功。
    good_update = client.put(
        f"/api/v1/admin/knowledge-bases/{kb_id}/acl",
        json={"expectedVersion": 1, "readers": [{"type": "USER", "userId": "user-admin-1"}]},
        headers={"X-CSRF-Token": csrf},
    )
    assert good_update.status_code == 200
    assert good_update.json()["rowVersion"] == 2


def test_document_upload_and_preview(client):
    csrf = get_auth_context(client, "admin", "admin123")

    # 第 1 步：创建知识库，作为文档导入与 ACL 测试的父资源。
    kb_resp = client.post(
        "/api/v1/admin/knowledge-bases",
        json={"name": "kb-doc-test", "readers": []},
        headers={"X-CSRF-Token": csrf},
    )
    kb_id = kb_resp.json()["id"]

    # 第 2 步：上传 Markdown 文档，触发完整 ingestion/build 流程。
    doc_content = (
        "# 规格说明\n\n"
        "知识库系统采用结构化切块策略。\n\n"
        "## 核心算法\n\n"
        "采用倒数排名融合（RRF）算法进行多路召回融合。\n\n"
        "```python\ndef test():\n    pass\n```\n"
    ).encode("utf-8")

    idem_key = "test-upload-key-01"
    upload_resp = client.post(
        "/api/v1/admin/documents",
        data={"kbId": kb_id, "buildConfig": "{}"},
        files={"file": ("spec.md", doc_content, "text/markdown")},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": idem_key},
    )
    assert upload_resp.status_code == 202
    upload_data = upload_resp.json()
    doc_id = upload_data["documentId"]
    version_id = upload_data["versionId"]
    job_id = upload_data["jobId"]
    assert upload_data["status"] == "QUEUED"

    # 第 3 步：持续轮询后台 job 到 COMPLETED，确认异步构建真正完成。
    job = None
    for _ in range(60):
        time.sleep(0.1)
        job_resp = client.get(f"/api/v1/admin/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        if job["status"] == "COMPLETED":
            break

    assert job is not None
    assert job["status"] == "COMPLETED"
    assert job["stage"] == "INDEXING"

    # 第 4 步：预览该文档版本，核对解析、SourceUnit 与 chunk 持久化结果。
    preview_resp = client.get(f"/api/v1/documents/{doc_id}/versions/{version_id}/preview")
    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert len(preview["units"]) > 0
    assert len(preview["chunks"]) > 0

    # SourceUnit 直接来自 Docling HybridChunker provenance；测试要验证适配层没有重新制造不一致的边界。
    kinds = {u["kind"] for u in preview["units"]}
    assert kinds == {"TEXT"}

    # 同时验证 parent 与 child chunk 存在，保证 parent-child 检索策略的数据基础完整。
    chunk_kinds = {c["kind"] for c in preview["chunks"]}
    assert "PARENT" in chunk_kinds
    assert "CHILD" in chunk_kinds

    # 第 5 步：上传新版本，验证版本递增与 active build 切换逻辑。
    ver2_content = "# 规格说明 V2\n\n更新了文档内容。\n".encode("utf-8")
    ver2_resp = client.post(
        f"/api/v1/admin/documents/{doc_id}/versions",
        data={"buildConfig": "{}"},
        files={"file": ("spec_v2.md", ver2_content, "text/markdown")},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "test-upload-v2-key"},
    )
    assert ver2_resp.status_code == 202
    v2_data = ver2_resp.json()
    assert v2_data["documentId"] == doc_id
    assert v2_data["versionId"] != version_id
