"""Milvus 混合检索与索引访问层。

同一个 canonical CHILD chunk 可以有 ORIGINAL / SUMMARY / QUESTION 多种 representation。
这些 representation 是“多种检索入口”，不是三篇独立文档，因此最终必须回到 canonical chunk 去重和融合，
否则一个 chunk 会因为拥有多个 representation 在 RRF 中重复投票。
"""
import hashlib
import os
import logging
import struct
import threading
from typing import List, Dict, Any, Optional
from pymilvus import DataType, Function, FunctionType, AnnSearchRequest, RRFRanker, CollectionSchema, FieldSchema
from sqlalchemy import select, and_, or_, exists
from app.config import settings, is_test_env
from app.db import get_db_connection
from app.errors import ApiException
from app.models import chunk_table, index_build_table, document_table, kb_grant_table, document_grant_table, representation_table
from app.retrieval.normalizer import normalize_terms
from app.retrieval.ranking import rrf_fuse
from app.retrieval.embedding import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks"


class MilvusHybridService:
    """负责 collection 初始化、representation upsert、删除和 BM25/Dense/Hybrid 查询。"""

    # 仅测试环境允许内存 mock；生产连接失败必须显式报错。
    # 本地单测使用内存索引兜底，不要求启动 Milvus 容器；生产路径仍走真实 Milvus，避免测试依赖外部服务。
    _mock_store: Dict[str, Dict[str, Any]] = {}
    _client = None
    _client_checked = False
    _client_lock = threading.Lock()

    @classmethod
    def get_client(cls):
        with cls._client_lock:
            return cls._connect_client()

    @classmethod
    def _connect_client(cls):
        if cls._client is None:
            if is_test_env() and not os.getenv("MILVUS_URI"):
                cls._client = None
                return None
            milvus_uri = os.getenv("MILVUS_URI") or f"http://{os.getenv('MILVUS_HOST', '127.0.0.1')}:{os.getenv('MILVUS_PORT', '19530')}"
            try:
                from pymilvus import MilvusClient
                client = MilvusClient(uri=milvus_uri, timeout=3.0)
                # collection 首次启动时创建：dense 用 HNSW+COSINE，sparse 由 Milvus BM25 Function 生成。
                if not client.has_collection(COLLECTION_NAME):
                    fields = [
                        FieldSchema("id", DataType.VARCHAR, max_length=128, is_primary=True),
                        FieldSchema("tenant_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("kb_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("document_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("version_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("build_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("index_epoch", DataType.INT64),
                        FieldSchema("chunk_id", DataType.VARCHAR, max_length=64),
                        FieldSchema("representation_type", DataType.VARCHAR, max_length=32),
                        FieldSchema("text_content", DataType.VARCHAR, max_length=65535, enable_analyzer=True),
                        FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
                        FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=1024),
                    ]
                    bm25_fn = Function(
                        name="text_bm25_emb",
                        function_type=FunctionType.BM25,
                        input_field_names=["text_content"],
                        output_field_names=["sparse_vector"],
                    )
                    schema = CollectionSchema(fields=fields, functions=[bm25_fn], auto_id=False, enable_dynamic_field=True)
                    index_params = client.prepare_index_params()
                    # HNSW 的 M/efConstruction 是建索引参数；查询时 num_candidates 会映射到 ef。
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type="HNSW",
                        metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200},
                    )
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25",
                    )
                    client.create_collection(
                        collection_name=COLLECTION_NAME,
                        schema=schema,
                        index_params=index_params,
                    )
                cls._client = client
                cls._client_checked = True
                logger.info("Connected to Milvus server at %s", milvus_uri)
            except Exception as e:
                if not is_test_env():
                    logger.error("Milvus connection failed at %s: %s", milvus_uri, e)
                else:
                    logger.warning("Milvus connection failed (%s); using in-memory mock store for test environment.", e)
                cls._client = None
        return cls._client

    @classmethod
    def upsert_chunks(
        cls,
        tenant_id: str,
        build_id: str,
        chunks: List[Dict[str, Any]],
        kb_id: str,
        doc_id: str,
        version_id: str,
        index_epoch: Optional[int] = None,
        representations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        run_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        if not chunks:
            return

        epoch = int(index_epoch or 0)
        batch_size = max(1, min(512, int(os.getenv("MILVUS_UPSERT_BATCH_SIZE", "128"))))
        client = cls.get_client()

        # 每个 child 至少有 ORIGINAL；SUMMARY/QUESTION 是同一 child 的派生表示。
        representations = representations or {}
        items: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk["id"])
            reps = representations.get(chunk_id) or [
                {"kind": "ORIGINAL", "ordinal": 0, "text_content": chunk["text_content"]}
            ]
            for rep in reps:
                kind = str(rep.get("kind") or "ORIGINAL").upper()
                if kind not in {"ORIGINAL", "SUMMARY", "QUESTION"}:
                    continue
                items.append(
                    {
                        "chunk": chunk,
                        "chunk_id": chunk_id,
                        "kind": kind,
                        "ordinal": int(rep.get("ordinal") or 0),
                        "text_content": str(rep.get("text_content") or ""),
                        "dense_vector": rep.get("dense_vector"),
                    }
                )

        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            # 入库链路可能已经批量生成向量；这里只为缺失 representation 补 embedding，避免重复收费。
            missing_indexes = [i for i, item in enumerate(batch) if not item.get("dense_vector")]
            generated = []
            if missing_indexes:
                generated = get_embeddings(
                    [batch[i]["text_content"] for i in missing_indexes],
                    dim=1024,
                    batch_size=batch_size,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    job_id=job_id,
                )
            generated_by_index = {
                idx: vector for idx, vector in zip(missing_indexes, generated)
            }
            records = []

            for batch_index, item in enumerate(batch):
                emb = item.get("dense_vector") or generated_by_index.get(batch_index)
                if emb is None:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "缺少 representation embedding")
                chunk = item["chunk"]
                chunk_id = item["chunk_id"]
                kind = item["kind"]
                ordinal = item["ordinal"]
                text_content = item["text_content"]
                norm = normalize_terms(text_content)
                record_key = f"{chunk_id}:{kind}:{ordinal}"
                if epoch > 0:
                    record_id = hashlib.sha256(
                        f"{tenant_id}:{build_id}:{epoch}:{record_key}".encode("utf-8")
                    ).hexdigest()
                else:
                    record_id = hashlib.sha256(
                        f"{tenant_id}:{build_id}:{record_key}".encode("utf-8")
                    ).hexdigest()
                record = {
                    "id": record_id,
                    "parent_id": str(chunk.get("parent_id") or ""),
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "document_id": doc_id,
                    "version_id": version_id,
                    "build_id": build_id,
                    "index_epoch": epoch,
                    "chunk_id": chunk_id,
                    "representation_type": kind,
                    "representation_ordinal": ordinal,
                    "text_content": text_content,
                    "normalized_terms": norm,
                    "dense_vector": emb,
                }
                if is_test_env():
                    cls._mock_store[record_id] = record
                records.append(record)

            if client:
                try:
                    client.upsert(collection_name=COLLECTION_NAME, data=records)
                except Exception as exc:
                    logger.error("Failed to upsert to Milvus: %s", exc)
                    if not is_test_env():
                        raise ApiException(503, "MILVUS_UNAVAILABLE", f"Milvus 入库失败: {exc}")
            elif not is_test_env():
                raise ApiException(503, "MILVUS_UNAVAILABLE", "Milvus 服务不可用，文档入库中止")

    @classmethod
    def delete_by_build(
        cls,
        tenant_id: str,
        build_id: str,
        index_epoch: Optional[int] = None,
    ):
        epoch = int(index_epoch) if index_epoch is not None else None
        to_delete = [
            key
            for key, value in cls._mock_store.items()
            if value.get("tenant_id") == tenant_id
            and value.get("build_id") == build_id
            and (epoch is None or int(value.get("index_epoch") or 0) == epoch)
        ]
        for key in to_delete:
            cls._mock_store.pop(key, None)

        client = cls.get_client()
        if client:
            try:
                filter_expr = f'tenant_id == "{tenant_id}" and build_id == "{build_id}"'
                if epoch is not None:
                    filter_expr += f" and index_epoch == {epoch}"
                client.delete(
                    collection_name=COLLECTION_NAME,
                    filter=filter_expr,
                )
            except Exception as e:
                logger.warning("Failed to delete from Milvus by build: %s", e)
                if not is_test_env():
                    raise ApiException(503, "MILVUS_UNAVAILABLE", f"Milvus 删除失败: {e}")

    @classmethod
    def delete_by_document(cls, tenant_id: str, doc_id: str):
        to_delete = [k for k, v in cls._mock_store.items() if v.get("tenant_id") == tenant_id and v.get("document_id") == doc_id]
        for k in to_delete:
            cls._mock_store.pop(k, None)

        client = cls.get_client()
        if client:
            try:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    filter=f'tenant_id == "{tenant_id}" and document_id == "{doc_id}"',
                )
            except Exception as e:
                logger.warning("Failed to delete from Milvus by document: %s", e)
                if not is_test_env():
                    raise ApiException(503, "MILVUS_UNAVAILABLE", f"Milvus 删除失败: {e}")

    @classmethod
    def search(
        cls,
        tenant_id: str,
        query: str,
        allowed_build_ids: List[str],
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None,
        mode: str = "HYBRID",
        rrf_enabled: bool = True,
        rrf_k: int = 60,
        allowed_build_epochs: Optional[Dict[str, int]] = None,
        representations: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        job_id: Optional[str] = None,
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # allowed_build_ids 已由权限层计算；为空时绝不能放宽为“搜索整个租户”。
        if not allowed_build_ids:
            return []

        mode = str(mode or "HYBRID").upper()
        if mode == "SPARSE":
            mode = "BM25"
        if mode not in {"DENSE", "BM25", "HYBRID"}:
            raise ApiException(409, "INVALID_RETRIEVAL_CONFIG", f"不支持的检索模式: {mode}")
        if mode == "HYBRID" and not rrf_enabled:
            raise ApiException(409, "INVALID_RETRIEVAL_CONFIG", "HYBRID 模式需要 RRF 融合")

        if top_k is not None:
            if dense_top_k is None:
                dense_top_k = top_k * 2
            if sparse_top_k is None:
                sparse_top_k = top_k * 2
            if fusion_top_k is None:
                fusion_top_k = top_k
        dense_top_k = 0 if dense_top_k is None and mode == "BM25" else (dense_top_k if dense_top_k is not None else 30)
        sparse_top_k = 0 if sparse_top_k is None and mode == "DENSE" else (sparse_top_k if sparse_top_k is not None else 30)
        fusion_top_k = fusion_top_k or 20
        # Dense HNSW 的搜索宽度至少覆盖 top_k，并设置 1000 上限控制延迟。
        num_candidates = max(
            max(dense_top_k, sparse_top_k, 1),
            min(1000, int(num_candidates or 200)),
        )

        # representation 只影响“如何命中”，下游会 hydrate 回 ORIGINAL 正文。
        representations = [str(value).upper() for value in (representations or ["ORIGINAL"])]
        if not representations or not set(representations).issubset({"ORIGINAL", "SUMMARY", "QUESTION"}):
            raise ApiException(409, "INVALID_RETRIEVAL_CONFIG", "representation 类型无效")
        rep_expr = ", ".join(f'"{value}"' for value in representations)

        output_fields = ["kb_id", "document_id", "version_id", "build_id", "index_epoch", "chunk_id", "representation_type", "text_content"]

        def normalize_result(res):
            """把 Milvus 返回结构统一成项目 hit，并在 canonical chunk 级去重。"""
            hits = []
            rows = res[0] if res and isinstance(res, list) and res and isinstance(res[0], list) else (res or [])
            for item in rows:
                entity = item.get("entity", {}) if isinstance(item, dict) else {}
                chunk_id = entity.get("chunk_id") or (item.get("id") if isinstance(item, dict) else None)
                hits.append({
                    "id": chunk_id,
                    "chunkId": chunk_id,
                    "kbId": entity.get("kb_id"),
                    "documentId": entity.get("document_id"),
                    "versionId": entity.get("version_id"),
                    "buildId": entity.get("build_id"),
                    "indexEpoch": int(entity.get("index_epoch") or 0),
                    "representationType": entity.get("representation_type") or "ORIGINAL",
                    "text": entity.get("text_content"),
                    "score": float(item.get("distance", item.get("score", 0.0)) or 0.0) if isinstance(item, dict) else 0.0,
                })
            # 同一 chunk 的 SUMMARY/QUESTION 可能同时命中；只保留该 route 中首次出现的 canonical chunk。
            deduped = []
            seen_chunks = set()
            for hit in hits:
                chunk_id = str(hit.get("chunkId") or "")
                if not chunk_id or chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                deduped.append(hit)
            return deduped

        client = cls.get_client()
        if client:
            try:
                # 新 build 使用 index_epoch 精确过滤，防止同 buildId 的旧索引代次残留被命中。
                if allowed_build_epochs:
                    grouped: Dict[int, List[str]] = {}
                    legacy_builds: List[str] = []
                    for allowed_build_id in allowed_build_ids:
                        expected_epoch = int(allowed_build_epochs.get(allowed_build_id, 0) or 0)
                        if expected_epoch > 0:
                            grouped.setdefault(expected_epoch, []).append(allowed_build_id)
                        else:
                            legacy_builds.append(allowed_build_id)

                    build_clauses = []
                    for epoch, build_ids in grouped.items():
                        ids_expr = ", ".join(f'"{build_id}"' for build_id in build_ids)
                        build_clauses.append(
                            f'(index_epoch == {epoch} and build_id in [{ids_expr}])'
                        )
                    if legacy_builds:
                        ids_expr = ", ".join(f'"{build_id}"' for build_id in legacy_builds)
                        build_clauses.append(f'build_id in [{ids_expr}]')

                    build_filter = " or ".join(build_clauses)
                    filter_expr = (
                        f'tenant_id == "{tenant_id}" and ({build_filter}) '
                        f'and representation_type in [{rep_expr}]'
                    )
                else:
                    build_ids_str = ", ".join(f'"{b}"' for b in allowed_build_ids)
                    filter_expr = (
                        f'tenant_id == "{tenant_id}" and build_id in [{build_ids_str}] '
                        f'and representation_type in [{rep_expr}]'
                    )

                # DENSE：先为 query 生成 embedding，再做 COSINE HNSW 查询。
                if mode == "DENSE":
                    query_emb = get_embeddings(
                        [query],
                        dim=1024,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        job_id=job_id,
                    )[0]
                    res = client.search(
                        collection_name=COLLECTION_NAME,
                        data=[query_emb],
                        anns_field="dense_vector",
                        search_params={"metric_type": "COSINE", "params": {"ef": num_candidates}},
                        limit=max(1, dense_top_k * max(1, len(representations))),
                        filter=filter_expr,
                        output_fields=output_fields,
                    )
                    return normalize_result(res)[:fusion_top_k]

                # BM25：直接把原 query 交给 Milvus sparse/BM25 字段。
                if mode == "BM25":
                    res = client.search(
                        collection_name=COLLECTION_NAME,
                        data=[query],
                        anns_field="sparse_vector",
                        search_params={"metric_type": "BM25"},
                        limit=max(1, sparse_top_k * max(1, len(representations))),
                        filter=filter_expr,
                        output_fields=output_fields,
                    )
                    return normalize_result(res)[:fusion_top_k]

                # HYBRID 故意不使用 Milvus 内部 hybrid RRF：
                # ORIGINAL/SUMMARY/QUESTION 若在去重前各算一票，同一个 chunk 会获得不公平的多重投票。
                # 所以先分别搜索 → canonical chunk 去重 → 再在应用层 RRF。
                # RRF 必须在 canonical chunk 层融合；如果直接在 Milvus representation 层做 hybrid RRF，
                # ORIGINAL/SUMMARY/QUESTION 会被当成独立文档，导致同一个 chunk 通过多个 representation 重复投票，
                # 去重发生得太晚就会人为放大该 chunk 的排名，因此先归并 representation 再融合。
                candidate_multiplier = max(1, len(representations))
                query_emb = get_embeddings(
                    [query],
                    dim=1024,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    job_id=job_id,
                )[0]
                dense_res = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_emb],
                    anns_field="dense_vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": num_candidates}},
                    limit=max(1, dense_top_k * candidate_multiplier),
                    filter=filter_expr,
                    output_fields=output_fields,
                )
                sparse_res = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query],
                    anns_field="sparse_vector",
                    search_params={"metric_type": "BM25"},
                    limit=max(1, sparse_top_k * candidate_multiplier),
                    filter=filter_expr,
                    output_fields=output_fields,
                )
                dense_hits = normalize_result(dense_res)[:dense_top_k]
                sparse_hits = normalize_result(sparse_res)[:sparse_top_k]
                return rrf_fuse(
                    [dense_hits, sparse_hits],
                    weights=[0.5, 0.5],
                    rrf_k=rrf_k,
                )[:fusion_top_k]
            except ApiException:
                raise
            except Exception as e:
                logger.error("Milvus search error: %s", e)
                if not is_test_env():
                    raise ApiException(503, "MILVUS_UNAVAILABLE", f"Milvus 检索失败: {e}")

        if not is_test_env():
            raise ApiException(503, "MILVUS_UNAVAILABLE", "Milvus 检索服务不可用")

        bm25_hits: List[Dict[str, Any]] = []
        dense_hits: List[Dict[str, Any]] = []
        query_terms = [t.lower() for t in query.split() if t.strip()]

        def record_is_allowed(item: Dict[str, Any]) -> bool:
            build_id = str(item.get("build_id") or "")
            if item.get("tenant_id") != tenant_id or build_id not in allowed_build_ids:
                return False
            if allowed_build_epochs:
                expected_epoch = int(allowed_build_epochs.get(build_id, 0) or 0)
                if expected_epoch > 0 and int(item.get("index_epoch") or 0) != expected_epoch:
                    return False
            return True

        if mode in {"BM25", "HYBRID"}:
            for item in cls._mock_store.values():
                if not record_is_allowed(item):
                    continue
                if item["representation_type"] not in representations:
                    continue
                text_lower = (item["text_content"] + " " + item["normalized_terms"]).lower()
                score = sum(1.0 for t in query_terms if t in text_lower)
                if score > 0:
                    bm25_hits.append({
                        "id": item["chunk_id"],
                        "kbId": item["kb_id"],
                        "documentId": item["document_id"],
                        "versionId": item["version_id"],
                        "buildId": item["build_id"],
                        "indexEpoch": int(item.get("index_epoch") or 0),
                        "chunkId": item["chunk_id"],
                        "text": item["text_content"],
                        "bm25_score": score,
                        "score": score,
                    })
            bm25_hits.sort(key=lambda x: x["bm25_score"], reverse=True)

        if mode in {"DENSE", "HYBRID"}:
            query_emb = get_embeddings(
                        [query],
                        dim=1024,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        job_id=job_id,
                    )[0]
            for item in cls._mock_store.values():
                if not record_is_allowed(item):
                    continue
                if item["representation_type"] not in representations:
                    continue
                v = item["dense_vector"]
                cos_sim = sum(a * b for a, b in zip(query_emb, v))
                dense_hits.append({
                    "id": item["chunk_id"],
                    "kbId": item["kb_id"],
                    "documentId": item["document_id"],
                    "versionId": item["version_id"],
                    "buildId": item["build_id"],
                    "indexEpoch": int(item.get("index_epoch") or 0),
                    "chunkId": item["chunk_id"],
                    "text": item["text_content"],
                    "dense_score": cos_sim,
                    "score": cos_sim,
                })
            dense_hits.sort(key=lambda x: x["dense_score"], reverse=True)

        def dedupe_canonical(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            output = []
            seen = set()
            for hit in items:
                chunk_id = str(hit.get("chunkId") or "")
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                output.append(hit)
            return output

        dense_hits = dedupe_canonical(dense_hits)
        bm25_hits = dedupe_canonical(bm25_hits)

        if mode == "DENSE":
            return dense_hits[:min(dense_top_k, fusion_top_k)]
        if mode == "BM25":
            return bm25_hits[:min(sparse_top_k, fusion_top_k)]

        fused = rrf_fuse(
            [dense_hits[:dense_top_k], bm25_hits[:sparse_top_k]],
            weights=[0.5, 0.5],
            rrf_k=rrf_k,
        )
        return fused[:fusion_top_k]

    @classmethod
    def rebuild_from_mysql(cls, tenant_id: str):
        """根据 MySQL 中当前 active build 幂等重建 Milvus 索引；重复执行不会产生重复向量，适用于灾难恢复和本地重建。"""
        with get_db_connection() as conn:
            stmt = (
                select(
                    document_table.c.kb_id,
                    document_table.c.id.label("document_id"),
                    document_table.c.active_build_id,
                    index_build_table.c.version_id,
                    index_build_table.c.fence,
                )
                .select_from(
                    document_table.join(
                        index_build_table,
                        and_(
                            index_build_table.c.tenant_id == document_table.c.tenant_id,
                            index_build_table.c.id == document_table.c.active_build_id,
                        ),
                    )
                )
                .where(
                    and_(
                        document_table.c.tenant_id == tenant_id,
                        document_table.c.status == "LIVE",
                        document_table.c.active_build_id.isnot(None),
                    )
                )
            )
            active_builds = conn.execute(stmt).fetchall()

            for b in active_builds:
                chunk_stmt = select(
                    chunk_table.c.id,
                    chunk_table.c.parent_id,
                    chunk_table.c.text_content,
                ).where(
                    and_(
                        chunk_table.c.tenant_id == tenant_id,
                        chunk_table.c.build_id == b.active_build_id,
                        chunk_table.c.kind == "CHILD",
                    )
                )
                chunks = conn.execute(chunk_stmt).fetchall()
                chunk_dicts = [
                    {
                        "id": str(c.id),
                        "parent_id": str(c.parent_id) if c.parent_id else None,
                        "text_content": c.text_content,
                    }
                    for c in chunks
                ]
                rep_rows = conn.execute(
                    select(
                        representation_table.c.chunk_id,
                        representation_table.c.kind,
                        representation_table.c.ordinal,
                        representation_table.c.text_content,
                        representation_table.c.vector_bytes,
                        representation_table.c.dims,
                    ).where(
                        and_(
                            representation_table.c.tenant_id == tenant_id,
                            representation_table.c.build_id == b.active_build_id,
                        )
                    )
                ).fetchall()
                representation_map: Dict[str, List[Dict[str, Any]]] = {}
                for rep in rep_rows:
                    item = {
                        "kind": str(rep.kind),
                        "ordinal": int(rep.ordinal or 0),
                        "text_content": str(rep.text_content or ""),
                    }
                    if rep.vector_bytes and rep.dims:
                        item["dense_vector"] = list(
                            struct.unpack(f"<{int(rep.dims)}f", bytes(rep.vector_bytes))
                        )
                    representation_map.setdefault(str(rep.chunk_id), []).append(item)

                if not is_test_env():
                    if chunk_dicts and not representation_map:
                        raise ApiException(
                            409,
                            "INDEX_REBUILD_REQUIRES_REINGEST",
                            "活动构建缺少持久化 representation，不能在无作业 owner 的重建流程中重新调用 Embedding；请重新入库",
                        )
                    if any(
                        not rep.get("dense_vector")
                        for reps in representation_map.values()
                        for rep in reps
                    ):
                        raise ApiException(
                            409,
                            "INDEX_REBUILD_REQUIRES_REINGEST",
                            "活动构建 representation 缺少持久化向量，请重新入库",
                        )

                cls.upsert_chunks(
                    tenant_id=tenant_id,
                    build_id=str(b.active_build_id),
                    chunks=chunk_dicts,
                    kb_id=str(b.kb_id),
                    doc_id=str(b.document_id),
                    version_id=str(b.version_id),
                    index_epoch=int(b.fence or 0),
                    representations=representation_map or None,
                )
