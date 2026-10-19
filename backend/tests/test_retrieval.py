import math
import uuid
from datetime import datetime, timezone
import pytest
import bcrypt

from app.models import (
    app_user_table,
    knowledge_base_table,
    document_table,
    index_build_table,
    chunk_table,
    kb_grant_table,
)
import app.db as app_db
from app.retrieval.normalizer import normalize_terms
from app.retrieval.ranking import rrf_fuse
from app.retrieval.embedding import generate_deterministic_embedding
from app.retrieval.milvus_service import MilvusHybridService


def test_normalizer_terms():
    text = "Error in backend/app/main.py: INVALID_MODEL_RESPONSE with active_build_id and expectedVersion"
    norm = normalize_terms(text)

    # 验证路径、snake_case、camelCase 与错误码都会生成可检索的归一化 token。
    assert "main" in norm.lower()
    assert "active" in norm.lower()
    assert "build" in norm.lower()
    assert "expected" in norm.lower()
    assert "version" in norm.lower()
    assert "invalid_model_response" in norm.lower()


def test_rrf_fusion():
    list1 = [{"id": "docA"}, {"id": "docB"}, {"id": "docC"}]
    list2 = [{"id": "docB"}, {"id": "docD"}, {"id": "docA"}]

    fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], rrf_k=60)
    assert len(fused) == 4
    # docB 在两个列表中排名分别为 2 和 1，RRF 累计分最高，因此融合后应排第一。
    assert fused[0]["id"] == "docB"
    assert fused[0]["score"] > fused[1]["score"]


def test_deterministic_embedding():
    vec = generate_deterministic_embedding("test text", dim=1024)
    assert len(vec) == 1024
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-4) == 1.0
    assert all(math.isfinite(x) for x in vec)


def test_hybrid_search_acl_and_rebuild(client):
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"
    viewer_id = "user-viewer-retrieval"
    hashed_pwd = bcrypt.hashpw(b"viewer123", bcrypt.gensalt(rounds=4)).decode("utf-8")

    with app_db.engine.begin() as conn:
        conn.execute(
            app_user_table.insert().values(
                tenant_id=tenant_id,
                id=viewer_id,
                login_name="viewer",
                password_hash=hashed_pwd,
                is_admin=False,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        )

        kb1_id = "kb-retrieval-1"
        kb2_id = "kb-retrieval-2"
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb1_id, name="KB 1", status="LIVE", row_version=1, created_at=now, updated_at=now))
        conn.execute(knowledge_base_table.insert().values(tenant_id=tenant_id, id=kb2_id, name="KB 2", status="LIVE", row_version=1, created_at=now, updated_at=now))

        # viewer 只授予 KB1 权限，用于验证检索不会跨 ACL 泄露 KB2 数据。
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb1_id,
                principal_type="USER",
                user_id=viewer_id,
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        # admin 同时授予 KB1/KB2 权限，作为完整可见性的对照组。
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb1_id,
                principal_type="USER",
                user_id="user-admin-1",
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=tenant_id,
                id=str(uuid.uuid4()),
                kb_id=kb2_id,
                principal_type="USER",
                user_id="user-admin-1",
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )

        # 文档 1 位于 KB1，对 viewer 可见。
        doc1_id = "doc-1"
        build1_id = "build-1"
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc1_id,
                kb_id=kb1_id,
                title="Public Doc in KB1",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build1_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=tenant_id,
                id=build1_id,
                document_id=doc1_id,
                version_id="ver-1",
                build_no=1,
                config_id="cfg-1",
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
                id="chunk-1",
                build_id=build1_id,
                parent_id=None,
                kind="CHILD",
                ordinal=1,
                text_content="Milvus hybrid search combines dense vector and BM25 text retrieval.",
                token_count=12,
                content_hash=b"hash1",
                created_at=now,
                updated_at=now,
            )
        )

        # 文档 2 位于 KB2，属于 viewer 不可见的敏感数据。
        doc2_id = "doc-2"
        build2_id = "build-2"
        conn.execute(
            document_table.insert().values(
                tenant_id=tenant_id,
                id=doc2_id,
                kb_id=kb2_id,
                title="Secret Doc in KB2",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build2_id,
                lifecycle_epoch=0,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=tenant_id,
                id=build2_id,
                document_id=doc2_id,
                version_id="ver-2",
                build_no=1,
                config_id="cfg-2",
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
                id="chunk-2",
                build_id=build2_id,
                parent_id=None,
                kind="CHILD",
                ordinal=1,
                text_content="Top secret milvus document retrieval only for admins.",
                token_count=10,
                content_hash=b"hash2",
                created_at=now,
                updated_at=now,
            )
        )

    # 根据 MySQL active build 重建测试索引，确保检索数据与持久化状态一致。
    MilvusHybridService.rebuild_from_mysql(tenant_id)

    # 第 1 步：以 viewer 登录，验证最小权限检索。
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "viewer", "password": "viewer123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_resp.status_code == 200

    # viewer 搜索只能看到 KB1 的 chunk-1，绝不能返回 KB2 的 chunk-2。
    search_resp = client.get("/api/v1/search?query=milvus")
    assert search_resp.status_code == 200
    evidence = search_resp.json()["evidence"]
    chunk_ids = [e["chunkId"] for e in evidence]
    assert "chunk-1" in chunk_ids
    assert "chunk-2" not in chunk_ids

    # 第 2 步：切换为 admin 登录，验证管理员可见范围。
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]
    login_admin = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "admin", "password": "admin123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_admin.status_code == 200

    # admin 搜索应同时看到 chunk-1 与 chunk-2，作为 ACL 过滤正确性的正向对照。
    admin_search = client.get("/api/v1/search?query=milvus")
    assert admin_search.status_code == 200
    admin_chunk_ids = [e["chunkId"] for e in admin_search.json()["evidence"]]
    assert "chunk-1" in admin_chunk_ids
    assert "chunk-2" in admin_chunk_ids

    # 第 3 步：按 build 删除索引数据，验证清理范围不会影响其他 build。
    MilvusHybridService.delete_by_build(tenant_id, build1_id)
    post_del_search = client.get("/api/v1/search?query=milvus")
    post_del_chunk_ids = [e["chunkId"] for e in post_del_search.json()["evidence"]]
    assert "chunk-1" not in post_del_chunk_ids
