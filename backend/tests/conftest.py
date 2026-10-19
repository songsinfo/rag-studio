import os
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import bcrypt

# 明确切换到测试环境，避免夹具误用开发/生产配置。
os.environ["RAG_ENVIRONMENT"] = "test"
os.environ["RAG_CSRF_SECRET"] = "test-secret-key-12345"
os.environ["LITELLM_BASE_URL"] = ""
os.environ["MILVUS_URI"] = ""
# Prevent the SDK from inferring release metadata from the CI runner's commit SHA.
os.environ["LANGFUSE_RELEASE"] = "synthetic-tests"

from app.config import settings
settings.ENVIRONMENT = "test"
settings.LITELLM_BASE_URL = ""
settings.MILVUS_URI = ""

from app.models import metadata, tenant_table, app_user_table, user_group_table, group_member_table, config_revision_table
import app.db as app_db
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    # 使用内存 SQLite 提供快速、可重复的测试数据库；每个测试会在受控夹具中重建状态。
    from app.identity.security import _login_attempts
    _login_attempts.clear()

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(test_engine)

    now = datetime.now(timezone.utc)
    hashed_pwd = bcrypt.hashpw(b"admin123", bcrypt.gensalt(rounds=4)).decode("utf-8")
    with test_engine.begin() as conn:
        conn.execute(
            tenant_table.insert().values(
                id="tenant-demo-1",
                code="demo",
                name="Demo Tenant",
                security_epoch=1,
                created_at=now,
            )
        )
        conn.execute(
            app_user_table.insert().values(
                tenant_id="tenant-demo-1",
                id="user-admin-1",
                login_name="admin",
                password_hash=hashed_pwd,
                is_admin=True,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            app_user_table.insert().values(
                tenant_id="tenant-demo-1",
                id="user-disabled-1",
                login_name="disabled_user",
                password_hash=hashed_pwd,
                is_admin=False,
                status="DISABLED",
                created_at=now,
                updated_at=now,
            )
        )
        for config_id, alias in {
            "cfg-model-chat-main": "chat-main",
            "cfg-model-embedding-main": "embedding-main",
            "cfg-model-judge-main": "judge-main",
            "cfg-model-rerank-main": "rerank-main",
        }.items():
            payload = {
                "modelAlias": alias,
                "currency": "CNY",
                "inputPricePerMillion": 0,
                "outputPricePerMillion": 0,
            }
            import hashlib, json
            conn.execute(
                config_revision_table.insert().values(
                    tenant_id="tenant-demo-1",
                    id=config_id,
                    kind="MODEL",
                    name=alias,
                    revision=1,
                    sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).digest(),
                    payload=payload,
                    status="PUBLISHED",
                    created_at=now,
                    updated_at=now,
                )
            )

        conn.execute(
            user_group_table.insert().values(
                tenant_id="tenant-demo-1",
                id="group-ops-1",
                name="Ops Group",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            group_member_table.insert().values(
                tenant_id="tenant-demo-1",
                id="gm-1",
                group_id="group-ops-1",
                user_id="user-admin-1",
                created_at=now,
                updated_at=now,
            )
        )

    # 将 app.db.engine 替换为测试引擎，确保被测代码和夹具操作的是同一个数据库。
    orig_engine = app_db.engine
    app_db.engine = test_engine

    # ingestion 测试保持 hermetic，不依赖真实 Docling 服务；生产环境仍始终调用真实 Docling。
    # 真实 HTTP 解析契约由下面的专门 parser 测试单独覆盖，避免单元测试与外部服务耦合。
    from app.knowledge.chunker import adapt_docling_hybrid_chunks
    import app.knowledge.service as knowledge_service

    def fake_parse_and_chunk_document(file_bytes, filename, build_config=None):
        try:
            text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = f"binary document {filename}"
        if not text:
            text = filename
        items = [{
            "filename": filename,
            "chunk_index": 0,
            "text": text,
            "raw_text": text,
            "num_tokens": max(1, len(text) // 4),
            "headings": [],
            "captions": [],
            "doc_items": [f"#/texts/{filename}"],
            "page_numbers": [1],
            "metadata": {},
        }]
        return adapt_docling_hybrid_chunks(items, parent_max_tokens=1200)

    monkeypatch.setattr(
        knowledge_service,
        "parse_and_chunk_document",
        fake_parse_and_chunk_document,
    )
    yield test_engine
    app_db.engine = orig_engine


@pytest.fixture
def client():
    return TestClient(app)
