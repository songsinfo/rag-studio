import json
import uuid
from datetime import datetime, timezone

import pytest

import app.db as app_db
from app.errors import ApiException
from app.identity.deps import Actor, assert_hits_still_readable
from app.models import (
    chunk_span_table,
    chunk_table,
    config_revision_table,
    document_table,
    index_build_table,
    kb_grant_table,
    knowledge_base_table,
    source_unit_table,
)
from app.retrieval.milvus_service import MilvusHybridService
from app.retrieval.pipeline import RetrievalPipeline


def test_config_driven_rerank_parent_expansion_and_provenance():
    MilvusHybridService._mock_store.clear()
    now = datetime.now(timezone.utc)
    tenant_id = "tenant-demo-1"
    kb_id = "kb-enterprise"
    doc_id = "doc-enterprise"
    build_id = "build-enterprise"
    version_id = "ver-enterprise"
    parent_id = "parent-enterprise"
    child_id = "child-enterprise"
    unit_id = "unit-enterprise"
    config_id = "cfg-enterprise-test"

    payload = {
        "retrieval": {
            "denseTopK": 20,
            "sparseTopK": 20,
            "fusionTopK": 10,
            "rrfK": 60,
            "finalTopK": 5,
        },
        "rerank": {
            "enabled": True,
            "topK": 5,
            "model": "rerank-main",
            "failOpen": False,
        },
        "parentExpansion": {"enabled": True, "maxChars": 6000},
    }

    with app_db.engine.begin() as conn:
        conn.execute(config_revision_table.insert().values(
            tenant_id=tenant_id, id=config_id, kind="RETRIEVAL", name="enterprise-test",
            revision=1, sha256=b"x" * 32, payload=payload, status="PUBLISHED",
            created_at=now, updated_at=now,
        ))
        conn.execute(knowledge_base_table.insert().values(
            tenant_id=tenant_id, id=kb_id, name="Enterprise KB", status="LIVE",
            row_version=1, created_at=now, updated_at=now,
        ))
        conn.execute(kb_grant_table.insert().values(
            tenant_id=tenant_id, id=str(uuid.uuid4()), kb_id=kb_id,
            principal_type="USER", user_id="user-admin-1", group_id=None,
            created_at=now, updated_at=now,
        ))
        conn.execute(document_table.insert().values(
            tenant_id=tenant_id, id=doc_id, kb_id=kb_id, title="Enterprise Doc",
            acl_mode="INHERIT", status="LIVE", active_build_id=build_id,
            lifecycle_epoch=0, row_version=1, created_at=now, updated_at=now,
        ))
        conn.execute(index_build_table.insert().values(
            tenant_id=tenant_id, id=build_id, document_id=doc_id, version_id=version_id,
            build_no=1, config_id="build-cfg", status="READY", capabilities={},
            fence=0, published_at=now, created_at=now, updated_at=now,
        ))
        conn.execute(source_unit_table.insert().values(
            tenant_id=tenant_id, id=unit_id, build_id=build_id, document_id=doc_id,
            unit_key="p_1", kind="TEXT", text_content="Milvus hybrid retrieval detail.",
            location={"page": 3}, machine_generated=False, artifact_id=None,
            content_hash=b"u" * 32, created_at=now, updated_at=now,
        ))
        conn.execute(chunk_table.insert().values(
            tenant_id=tenant_id, id=parent_id, build_id=build_id, parent_id=None,
            kind="PARENT", ordinal=1,
            text_content="Parent context explains Milvus hybrid retrieval with permissions and evidence.",
            token_count=20, content_hash=b"p" * 32, created_at=now, updated_at=now,
        ))
        conn.execute(chunk_table.insert().values(
            tenant_id=tenant_id, id=child_id, build_id=build_id, parent_id=parent_id,
            kind="CHILD", ordinal=1, text_content="Milvus hybrid retrieval.",
            token_count=5, content_hash=b"c" * 32, created_at=now, updated_at=now,
        ))
        conn.execute(chunk_span_table.insert().values(
            tenant_id=tenant_id, id=str(uuid.uuid4()), build_id=build_id,
            chunk_id=child_id, unit_id=unit_id, ordinal=1, start_cp=0, end_cp=24,
            location={"page": 3}, created_at=now, updated_at=now,
        ))

    MilvusHybridService.upsert_chunks(
        tenant_id=tenant_id,
        build_id=build_id,
        chunks=[{"id": child_id, "parent_id": parent_id, "text_content": "Milvus hybrid retrieval."}],
        kb_id=kb_id,
        doc_id=doc_id,
        version_id=version_id,
    )

    actor = Actor(tenant_id=tenant_id, user_id="user-admin-1", admin=True, security_epoch=1)
    result = RetrievalPipeline.retrieve(
        actor=actor,
        query="Milvus",
        config_id=config_id,
        run_id=None,
    )

    assert result["config"]["config_id"] == config_id
    assert result["hits"]
    hit = result["hits"][0]
    assert hit["buildId"] == build_id
    assert hit["versionId"] == version_id
    assert hit["parentChunkId"] == parent_id
    assert hit["matchedText"] == "Milvus hybrid retrieval."
    assert hit["text"].startswith("Parent context explains")
    assert hit["sourceUnits"][0]["unitId"] == unit_id
    assert hit["sourceUnits"][0]["location"]["page"] == 3
    assert hit["rerankScore"] is not None

    with app_db.engine.begin() as conn:
        conn.execute(document_table.update().where(document_table.c.id == doc_id).values(status="TRASHED"))
    with app_db.engine.connect() as conn:
        with pytest.raises(ApiException) as exc:
            assert_hits_still_readable(conn, actor, result["hits"])
    assert exc.value.code == "SOURCE_CHANGED"


def test_milvus_index_epoch_isolates_stale_attempt_cleanup():
    MilvusHybridService._mock_store.clear()
    common = {
        "tenant_id": "tenant-demo-1",
        "build_id": "build-fenced",
        "kb_id": "kb-fenced",
        "doc_id": "doc-fenced",
        "version_id": "ver-fenced",
    }

    MilvusHybridService.upsert_chunks(
        **common,
        chunks=[{"id": "child-old", "parent_id": None, "text_content": "Milvus stale attempt"}],
        index_epoch=1,
    )
    MilvusHybridService.upsert_chunks(
        **common,
        chunks=[{"id": "child-winner", "parent_id": None, "text_content": "Milvus winning attempt"}],
        index_epoch=2,
    )

    # 模拟旧 epoch=1 worker 在 epoch=2 已发布后才恢复执行清理，验证僵尸 worker 场景。
    # 旧 worker 的删除操作必须物理限定在自己的 generation，绝不能误删新 epoch 的索引数据。
    MilvusHybridService.delete_by_build(
        common["tenant_id"],
        common["build_id"],
        index_epoch=1,
    )

    remaining = list(MilvusHybridService._mock_store.values())
    assert len(remaining) == 1
    assert remaining[0]["chunk_id"] == "child-winner"
    assert remaining[0]["index_epoch"] == 2

    hits = MilvusHybridService.search(
        tenant_id=common["tenant_id"],
        query="Milvus",
        allowed_build_ids=[common["build_id"]],
        allowed_build_epochs={common["build_id"]: 2},
        mode="BM25",
        sparse_top_k=5,
        fusion_top_k=5,
    )
    assert [h["chunkId"] for h in hits] == ["child-winner"]
    assert hits[0]["indexEpoch"] == 2


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("guide.pdf", "application/pdf"),
        ("guide.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("guide.md", "text/markdown"),
        ("guide.txt", "text/plain"),
    ],
)
def test_all_document_formats_use_docling_hybrid_chunker(monkeypatch, filename, content_type):
    from app.knowledge import parser

    calls = []

    class FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {
                "chunks": [
                    {
                        "filename": filename,
                        "chunk_index": 0,
                        "text": "PDF 标题\n来自 Docling HybridChunker 的正文。",
                        "raw_text": "来自 Docling HybridChunker 的正文。",
                        "num_tokens": 18,
                        "headings": ["PDF 标题"],
                        "captions": [],
                        "doc_items": ["#/texts/0"],
                        "page_numbers": [1],
                        "metadata": {},
                    }
                ],
                "documents": [],
                "processing_time": 0.01,
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setenv("DOCLING_URL", "http://docling:5001")
    monkeypatch.setattr(parser.httpx, "Client", FakeClient)

    units, chunks, spans = parser.parse_and_chunk_document(b"fake-content", filename)
    assert calls
    url, kwargs = calls[0]
    assert url.endswith("/v1/chunk/hybrid/file")
    assert kwargs["files"]["files"][2] == content_type
    assert kwargs["data"]["chunking_max_tokens"] == "400"
    assert kwargs["data"]["chunking_merge_peers"] == "true"
    assert kwargs["data"]["chunking_include_raw_text"] == "true"
    assert kwargs["data"]["convert_do_table_structure"] == "true"

    assert len(units) == 1
    assert units[0]["kind"] == "TEXT"
    assert units[0]["location"]["headings"] == ["PDF 标题"]
    children = [c for c in chunks if c["kind"] == "CHILD"]
    assert len(children) == 1
    assert children[0]["token_count"] == 18
    assert "Docling HybridChunker" in children[0]["text_content"]
    assert spans[0]["unit_id"] == units[0]["id"]
