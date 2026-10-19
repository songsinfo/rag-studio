"""检索主流水线。

主流程：
1. 读取并校验检索配置；
2. 根据当前 Actor/评估快照计算真正可读的 build；
3. 可选 Rewrite / Multi-Query / HyDE 生成检索查询；
4. BM25 与 Dense 分路召回，再用 RRF 融合；
5. 可选 Rerank、MMR、Parent Expansion；
6. 应用上下文预算并补齐 provenance；
7. 返回前再次做权限校验。

最重要的安全边界：任何候选文本在离开应用进程交给外部模型（reranker/Judge/LLM）前，
都必须重新确认当前仍可读，避免“检索时可读、发送时已撤权”的 TOCTOU 问题。
"""
import json
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select

from app.db import get_db_connection
from app.identity.deps import Actor, assert_hits_still_readable, get_readable_build_records
from app.models import chunk_span_table, chunk_table, source_unit_table, representation_table
from app.retrieval.config import RetrievalConfig, load_retrieval_config
from app.retrieval.milvus_service import MilvusHybridService
from app.retrieval.reranker import rerank_hits
from app.retrieval.ranking import rrf_fuse
from app.retrieval.embedding import get_embeddings
from app.qa.model_gateway import ModelGateway
from app.errors import ApiException

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """把多路检索、融合、重排、去冗余和上下文扩展串成一个稳定入口。"""

    @classmethod
    def retrieve(
        cls,
        actor: Actor,
        query: str,
        *,
        kb_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        config_id: Optional[str] = None,
        run_id: Optional[str] = None,
        job_id: Optional[str] = None,
        limit_override: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        diagnostic_fn=None,
    ) -> Dict[str, Any]:
        # Evaluation-only observer; exceptions cannot alter ranking or success.
        def capture(stage, candidates, limit, route_index=None):
            if diagnostic_fn is not None:
                try:
                    with get_db_connection() as conn:
                        assert_hits_still_readable(conn, actor, candidates, kb_ids, document_ids)
                    diagnostic_fn(stage, candidates, limit, route_index)
                except Exception:
                    logger.warning("eval_pipeline_diagnostic_skipped")

        # 所有后续阶段只使用已经归一化/校验过的 RetrievalConfig。
        cfg = load_retrieval_config(actor.tenant_id, config_id)

        with get_db_connection() as conn:
            if run_id:
                # 评估运行可能锁定固定 document/build 范围；在线检索则没有这个额外约束。
                from app.qa.context import run_document_scope
                run_scope = run_document_scope(conn, actor, run_id)
                if run_scope is not None:
                    document_ids = run_scope if document_ids is None else list(set(document_ids).intersection(run_scope))
            # 这里得到的是“当前 Actor 真正可读且生效”的 build，而不是仅按请求参数过滤。
            readable = get_readable_build_records(conn, actor, kb_ids, document_ids)

        # MMR 需要原文向量计算文档间相似度，因此即使主检索是 BM25，也要求 dense 能力。
        needs_dense = cfg.mode in {"DENSE", "HYBRID"} or cfg.mmr_enabled
        if needs_dense:
            for row in readable:
                dims = row.get("embeddingDims")
                fingerprint = row.get("modelFingerprint")
                if (dims is not None and dims != 1024) or (
                    fingerprint is not None and fingerprint != "embedding-main"
                ):
                    raise ApiException(
                        409,
                        "INDEX_CAPABILITY_MISSING",
                        "当前生效构建的 Embedding 模型或维度与检索配置不兼容，请重新构建索引",
                    )
                capabilities = row.get("capabilities") or {}
                available_reps = set(capabilities.get("representations") or ["ORIGINAL"])
                if not set(cfg.representations).issubset(available_reps):
                    raise ApiException(
                        409,
                        "INDEX_CAPABILITY_MISSING",
                        "当前生效构建缺少检索所需 representation，请重新构建索引",
                    )

        # 后续 Milvus 查询只允许命中这些 build，ACL/版本边界在这里转成检索层过滤条件。
        allowed_build_ids = [row["buildId"] for row in readable]
        allowed_build_epochs = {
            row["buildId"]: int(row.get("indexEpoch") or 0)
            for row in readable
        }
        # “没有权限范围”是正常的空结果，不应该继续调用模型或向量服务。
        if not allowed_build_ids:
            capture("scope", [], 0)
            return {
                "hits": [],
                "warnings": [],
                "config": cfg.as_dict(),
                "allowedBuildCount": 0,
            }

        warnings: List[str] = []
        # independent_query 用于后续 rerank/MMR；query_variants 用于多路召回；HyDE 只进 dense 路。
        independent_query, query_variants, hyde_text = cls._prepare_queries(
            actor=actor,
            query=query,
            history=history or [],
            cfg=cfg,
            run_id=run_id,
            job_id=job_id,
            warnings=warnings,
        )

        # BM25 与 Dense 视为独立 route：每个 query variant 都可以形成一条排名列表。
        # 与 Java 基线保持相同路由语义：BM25 变体和 dense 变体分别形成独立的 RRF 输入路线，
        # HyDE 只贡献给 dense 家族，不能让同一个改写结果同时在 sparse/dense 两边重复投票。
        ranked_lists: List[List[Dict[str, Any]]] = []
        route_weights: List[float] = []
        use_sparse = cfg.mode in {"BM25", "HYBRID"}
        use_dense = cfg.mode in {"DENSE", "HYBRID"}

        if use_sparse:
            # Hybrid 时 sparse 家族总权重占 0.5，再平均分给每个 query variant。
            sparse_weight = (0.5 if use_dense else 1.0) / max(1, len(query_variants))
            for route_index, query_variant in enumerate(query_variants):
                route = MilvusHybridService.search(
                    tenant_id=actor.tenant_id,
                    query=query_variant,
                    allowed_build_ids=allowed_build_ids,
                    dense_top_k=0,
                    sparse_top_k=cfg.sparse_top_k,
                    fusion_top_k=cfg.sparse_top_k,
                    mode="BM25",
                    rrf_enabled=False,
                    rrf_k=cfg.rrf_k,
                    allowed_build_epochs=allowed_build_epochs,
                    representations=list(cfg.representations),
                    run_id=run_id,
                    job_id=job_id,
                    num_candidates=cfg.num_candidates,
                )
                capture("bm25_recall", route, cfg.sparse_top_k, route_index)
                if route:
                    ranked_lists.append(route)
                    route_weights.append(sparse_weight)

        if use_dense:
            dense_queries = list(query_variants)
            # HyDE 的“假想答案段落”只用于向量相似度，不参与 BM25 关键词检索。
            if hyde_text:
                dense_queries.append(hyde_text)
            dense_weight = (0.5 if use_sparse else 1.0) / max(1, len(dense_queries))
            for route_index, dense_query in enumerate(dense_queries):
                route = MilvusHybridService.search(
                    tenant_id=actor.tenant_id,
                    query=dense_query,
                    allowed_build_ids=allowed_build_ids,
                    dense_top_k=cfg.dense_top_k,
                    sparse_top_k=0,
                    fusion_top_k=cfg.dense_top_k,
                    mode="DENSE",
                    rrf_enabled=False,
                    rrf_k=cfg.rrf_k,
                    allowed_build_epochs=allowed_build_epochs,
                    representations=list(cfg.representations),
                    run_id=run_id,
                    job_id=job_id,
                    num_candidates=cfg.num_candidates,
                )
                capture("dense_recall", route, cfg.dense_top_k, route_index)
                if route:
                    ranked_lists.append(route)
                    route_weights.append(dense_weight)

        # 只有多路结果时才做 RRF；单路直接保留原排名，避免无意义地重新打分。
        if not ranked_lists:
            hits = []
        elif len(ranked_lists) == 1:
            hits = ranked_lists[0][:cfg.fusion_top_k]
        else:
            hits = rrf_fuse(
                ranked_lists,
                weights=route_weights,
                rrf_k=cfg.rrf_k,
            )[:cfg.fusion_top_k]

        capture("rrf_fusion", hits, cfg.fusion_top_k)
        # SUMMARY/QUESTION 等 representation 只用于匹配；真正给模型看的正文恢复成 ORIGINAL。
        hits = cls._hydrate_original_chunks(actor.tenant_id, hits)

        # Reranker 是外部模型边界：候选文本离开应用进程前必须重新鉴权。
        # 这是专门防止“检索后撤权”的竞态窗口。
        # rerank 属于外部模型边界；候选文本离开应用进程前必须逐条重新检查授权，
        # 防止检索后 ACL 被撤销时仍把已失权文本发送给 reranker。
        with get_db_connection() as conn:
            assert_hits_still_readable(conn, actor, hits, kb_ids, document_ids)

        capture("hydrate_and_acl", hits, cfg.fusion_top_k)

        if cfg.rerank_enabled and hits:
            hits, rerank_warnings = rerank_hits(
                tenant_id=actor.tenant_id,
                run_id=run_id,
                query=independent_query,
                job_id=job_id,
                hits=hits,
                model_alias=cfg.rerank_model,
                top_k=cfg.rerank_top_k,
                fail_open=cfg.rerank_fail_open,
            )
            warnings.extend(rerank_warnings)

        capture("rerank", hits, cfg.fusion_top_k)
        # MMR 放在 rerank 之后：先保证相关性，再从高质量候选里压低语义重复。
        if cfg.mmr_enabled and hits:
            hits = cls._mmr_select(
                actor.tenant_id,
                independent_query,
                hits,
                cfg.final_top_k,
                cfg.mmr_lambda,
                run_id=run_id,
                job_id=job_id,
            )

        capture("mmr", hits, cfg.fusion_top_k)
        # Parent Expansion 最后做，避免大段 parent 文本影响前面的 rerank/MMR 排序。
        if cfg.parent_expansion_enabled and hits:
            hits = cls._expand_parents(actor.tenant_id, hits, cfg, warnings)

        capture("parent_expansion", hits, cfg.fusion_top_k)
        # Parent 扩展可能让文本突然变大，所以必须在扩展后执行最终上下文预算。
        # parent expansion 完成后再应用 Java 基线的 context budget；预算针对最终送入模型的上下文，而不是原始 child。
        budgeted_hits: List[Dict[str, Any]] = []
        used_bytes = 0
        for hit in hits:
            text_value = str(hit.get("text") or "")
            size = len(text_value.encode("utf-8"))
            if used_bytes + size > cfg.context_max_bytes:
                if not budgeted_hits:
                    # 最佳 parent 自己就超预算时，尝试退回它的 canonical child。
                    # 这样既避免上下文为空，也不粗暴截断原文导致引用位置失真。
                    # 即使 parent expansion 后首条就超预算，也至少保留最佳命中；
                    # 如果展开后的 parent 单条过大，则退回 canonical child 文本，避免因为预算裁剪导致结果完全为空。
                    child_id = str(hit.get("chunkId") or "")
                    fallback = cls._hydrate_original_chunks(
                        actor.tenant_id,
                        [{"chunkId": child_id, **hit}],
                    )[0] if child_id else hit
                    fallback_text = str(fallback.get("text") or "")
                    if len(fallback_text.encode("utf-8")) <= cfg.context_max_bytes:
                        budgeted_hits.append(fallback)
                warnings.append("CONTEXT_TRIMMED")
                break
            budgeted_hits.append(hit)
            used_bytes += size
            if len(budgeted_hits) >= cfg.final_top_k:
                break
        hits = budgeted_hits
        capture("context_budget", hits, cfg.final_top_k)

        # 排名已经稳定后再补 provenance，避免前面每个阶段都携带大块定位元数据。
        hits = cls._attach_provenance(actor.tenant_id, hits)
        final_limit = cfg.final_top_k
        if limit_override is not None:
            final_limit = max(1, min(50, int(limit_override)))
        hits = hits[:final_limit]

        # 返回给上游前最后一次鉴权：后续 QA 会把这些文本发送给生成模型。
        with get_db_connection() as conn:
            assert_hits_still_readable(conn, actor, hits, kb_ids, document_ids)

        capture("finalize_hits", hits, final_limit)
        return {
            "hits": hits,
            "warnings": warnings,
            "config": cfg.as_dict(),
            "allowedBuildCount": len(allowed_build_ids),
        }

    @staticmethod
    def _hydrate_original_chunks(tenant_id: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把命中的派生 representation 恢复为 canonical ORIGINAL chunk 正文。"""
        chunk_ids = [str(hit.get("chunkId")) for hit in hits if hit.get("chunkId")]
        if not chunk_ids:
            return hits
        with get_db_connection() as conn:
            rows = conn.execute(
                select(chunk_table.c.id, chunk_table.c.text_content).where(
                    and_(
                        chunk_table.c.tenant_id == tenant_id,
                        chunk_table.c.id.in_(chunk_ids),
                        chunk_table.c.kind == "CHILD",
                    )
                )
            ).fetchall()
        original = {str(row.id): str(row.text_content or "") for row in rows}
        hydrated = []
        for hit in hits:
            item = dict(hit)
            chunk_id = str(item.get("chunkId") or "")
            if item.get("representationType") not in {None, "ORIGINAL"}:
                item["matchedRepresentationText"] = str(item.get("text") or "")
            if chunk_id in original:
                item["text"] = original[chunk_id]
            hydrated.append(item)
        return hydrated

    @staticmethod
    def _prepare_queries(
        actor: Actor,
        query: str,
        history: List[Dict[str, str]],
        cfg: RetrievalConfig,
        run_id: Optional[str],
        job_id: Optional[str],
        warnings: List[str],
    ) -> Tuple[str, List[str], Optional[str]]:
        original = query.strip()
        # independent 是“脱离历史也能独立理解”的查询；默认就是原问题。
        independent = original
        variants = [original]
        # Rewrite 只在存在历史时有意义，用来把“那它呢？”这类追问改成独立检索问题。
        if cfg.rewrite_enabled and history:
            messages = [
                {
                    "role": "system",
                    "content": "结合历史对话把当前追问改写成不依赖上下文的独立检索问题。不得添加新事实。只输出 JSON {\"query\":\"...\"}。",
                },
                {
                    "role": "user",
                    "content": json.dumps({"history": history[-8:], "question": query}, ensure_ascii=False),
                },
            ]
            try:
                raw = ModelGateway.chat(
                    actor.tenant_id,
                    run_id,
                    "REWRITE",
                    messages,
                    max_tokens=300,
                    model_alias="chat-main",
                    job_id=job_id,
                )
                try:
                    parsed = json.loads(raw)
                except Exception as exc:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Query Rewrite 未返回合法 JSON") from exc
                rewritten = str(parsed.get("query", "")).strip()
                if not rewritten or len(rewritten) > 2000:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Query Rewrite 返回内容无效")
                independent = rewritten
                if rewritten not in variants and len(variants) < cfg.max_variants:
                    variants.append(rewritten)
            except ApiException as exc:
                fallback_codes = {
                    "MODEL_GATEWAY_UNAVAILABLE",
                    "MODEL_TIMEOUT",
                    "MODEL_PROVIDER_ERROR",
                    "MODEL_RETRYABLE",
                    "INVALID_MODEL_OUTPUT",
                }
                # 只有明确允许 fallback 且错误属于模型临时/格式问题时，才退回原查询。
                # 权限、预算等业务错误不能被“降级”掩盖。
                if not cfg.allow_rewrite_fallback or exc.code not in fallback_codes:
                    raise
                warnings.append("REWRITE_SKIPPED")

        # Multi-Query 生成等价问法提升召回，但不得改变原问题约束。
        if cfg.multi_query_enabled:
            raw = ModelGateway.chat(
                actor.tenant_id,
                run_id,
                "MULTI_QUERY",
                [
                    {
                        "role": "system",
                        "content": "为检索生成最多两个不改变约束条件的等价问法。只输出 JSON {\"queries\":[\"...\"]}。",
                    },
                    {"role": "user", "content": independent},
                ],
                max_tokens=400,
                model_alias="chat-main",
                job_id=job_id,
            )
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                raise ApiException(502, "INVALID_MODEL_OUTPUT", "Multi-Query 未返回合法 JSON") from exc
            queries = parsed.get("queries")
            if not isinstance(queries, list) or len(queries) > 2:
                raise ApiException(502, "INVALID_MODEL_OUTPUT", "Multi-Query 返回数量无效")
            for item in queries:
                value = str(item).strip()
                if not value or len(value) > 2000:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Multi-Query 返回内容无效")
                if value not in variants:
                    variants.append(value)
                if len(variants) >= cfg.max_variants:
                    break

        hyde_text: Optional[str] = None
        # HyDE 生成“假想资料段落”作为 dense query；它不是事实，绝不能直接展示给用户。
        if cfg.hyde_enabled:
            hyde_text = ModelGateway.chat(
                actor.tenant_id,
                run_id,
                "HYDE",
                [
                    {
                        "role": "system",
                        "content": "生成一个不超过300 token的假想资料段落，只用于向量检索。不得声称其为事实。",
                    },
                    {"role": "user", "content": independent},
                ],
                max_tokens=300,
                model_alias="chat-main",
                job_id=job_id,
            ).strip()
            if not hyde_text:
                raise ApiException(502, "INVALID_MODEL_OUTPUT", "HyDE 返回内容为空")

        return independent, variants, hyde_text

    @staticmethod
    def _mmr_select(
        tenant_id: str,
        query: str,
        hits: List[Dict[str, Any]],
        count: int,
        lambda_value: float,
        run_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # 0/1 个候选没有多样性选择空间，直接返回。
        if len(hits) <= 1:
            return hits[:count]
        chunk_ids = [str(hit.get("chunkId") or "") for hit in hits]
        with get_db_connection() as conn:
            rows = conn.execute(
                select(
                    representation_table.c.chunk_id,
                    representation_table.c.vector_bytes,
                    representation_table.c.dims,
                ).where(
                    and_(
                        representation_table.c.tenant_id == tenant_id,
                        representation_table.c.chunk_id.in_(chunk_ids),
                        representation_table.c.kind == "ORIGINAL",
                        representation_table.c.vector_bytes.isnot(None),
                    )
                )
            ).fetchall()
        vector_map: Dict[str, List[float]] = {}
        for row in rows:
            dims = int(row.dims or 0)
            raw = bytes(row.vector_bytes or b"")
            if dims <= 0 or len(raw) != dims * 4:
                continue
            vector_map[str(row.chunk_id)] = list(struct.unpack(f"<{dims}f", raw))
        # MMR 必须基于同一套 ORIGINAL embedding；缺任何一个都不能悄悄跳过，否则结果不可解释。
        if any(chunk_id not in vector_map for chunk_id in chunk_ids):
            raise ApiException(409, "INDEX_CAPABILITY_MISSING", "MMR 缺少原文向量，请重新构建索引")
        doc_vectors = [vector_map[chunk_id] for chunk_id in chunk_ids]

        def cosine(left: List[float], right: List[float]) -> float:
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(a * a for a in left))
            right_norm = math.sqrt(sum(b * b for b in right))
            if left_norm == 0.0 or right_norm == 0.0:
                return 0.0
            value = dot / (left_norm * right_norm)
            return max(-1.0, min(1.0, value))

        # 先把原相关性分数 min-max 归一化到 0..1，才能与 0..1 的冗余相似度组合。
        raw_scores = [float(hit.get("score", 0.0) or 0.0) for hit in hits]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        selected: List[int] = []
        remaining = list(range(len(hits)))
        target = min(max(1, count), len(hits))
        while remaining and len(selected) < target:
            best_index = None
            best_score = float("-inf")
            for idx in remaining:
                relevance = (
                    1.0
                    if max_score == min_score
                    else (raw_scores[idx] - min_score) / (max_score - min_score)
                )
                # 冗余取“与已选集合中最相似的那一个”，避免连续选出近重复段落。
                redundancy = max(
                    (
                        (cosine(doc_vectors[idx], doc_vectors[chosen]) + 1.0) / 2.0
                        for chosen in selected
                    ),
                    default=0.0,
                )
                # λ 越大越偏相关性，越小越强调多样性。
                score = lambda_value * relevance - (1.0 - lambda_value) * redundancy
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_index is None:
                break
            selected.append(best_index)
            remaining.remove(best_index)

        return [hits[index] for index in selected]

    @staticmethod
    def _expand_parents(
        tenant_id: str,
        hits: List[Dict[str, Any]],
        cfg: RetrievalConfig,
        warnings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        warnings = warnings if warnings is not None else []
        child_ids = [str(h.get("chunkId")) for h in hits if h.get("chunkId")]
        if not child_ids:
            return hits

        with get_db_connection() as conn:
            child_rows = conn.execute(
                select(
                    chunk_table.c.id,
                    chunk_table.c.build_id,
                    chunk_table.c.parent_id,
                    chunk_table.c.text_content,
                ).where(
                    and_(
                        chunk_table.c.tenant_id == tenant_id,
                        chunk_table.c.id.in_(child_ids),
                        chunk_table.c.kind == "CHILD",
                    )
                )
            ).fetchall()
            child_map = {str(r.id): r for r in child_rows}
            parent_ids = [str(r.parent_id) for r in child_rows if r.parent_id]
            parent_rows = []
            if parent_ids:
                parent_rows = conn.execute(
                    select(
                        chunk_table.c.id,
                        chunk_table.c.build_id,
                        chunk_table.c.text_content,
                    ).where(
                        and_(
                            chunk_table.c.tenant_id == tenant_id,
                            chunk_table.c.id.in_(parent_ids),
                            chunk_table.c.kind == "PARENT",
                        )
                    )
                ).fetchall()

        parent_map = {str(r.id): r for r in parent_rows}
        expanded: List[Dict[str, Any]] = []
        # 多个 child 可能属于同一个 parent；同一 parent 只保留一次，避免上下文重复。
        seen_contexts = set()

        for hit in hits:
            item = dict(hit)
            child = child_map.get(str(hit.get("chunkId")))
            context_key: Tuple[str, str]
            if child and child.parent_id:
                parent = parent_map.get(str(child.parent_id))
                if parent and str(parent.build_id) == str(hit.get("buildId")):
                    parent_text = str(parent.text_content or "")
                    # parent 过大时不截断拼接，而是保留 child，避免大段上下文挤掉其他证据。
                    if len(parent_text) <= cfg.parent_max_chars:
                        item["matchedText"] = str(hit.get("text") or "")
                        item["parentChunkId"] = str(parent.id)
                        item["text"] = parent_text
                        item["parentTruncated"] = False
                        context_key = (str(hit.get("buildId")), str(parent.id))
                    else:
                        warnings.append("PARENT_TRIMMED")
                        item["parentTruncated"] = True
                        context_key = (str(hit.get("buildId")), str(hit.get("chunkId")))
                else:
                    context_key = (str(hit.get("buildId")), str(hit.get("chunkId")))
            else:
                context_key = (str(hit.get("buildId")), str(hit.get("chunkId")))

            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)
            expanded.append(item)

        return expanded

    @staticmethod
    def _attach_provenance(tenant_id: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把最终命中映射回 SourceUnit/span，供引用展示和权限追踪使用。"""
        original_chunk_ids = [str(h.get("chunkId")) for h in hits if h.get("chunkId")]
        parent_ids = [str(h.get("parentChunkId")) for h in hits if h.get("parentChunkId")]
        if not original_chunk_ids and not parent_ids:
            return hits

        with get_db_connection() as conn:
            parent_children: Dict[str, List[str]] = {}
            child_ordinals: Dict[str, int] = {}
            if parent_ids:
                rows = conn.execute(
                    select(
                        chunk_table.c.id,
                        chunk_table.c.parent_id,
                        chunk_table.c.ordinal,
                    ).where(
                        and_(
                            chunk_table.c.tenant_id == tenant_id,
                            chunk_table.c.parent_id.in_(parent_ids),
                            chunk_table.c.kind == "CHILD",
                        )
                    ).order_by(chunk_table.c.ordinal.asc())
                ).fetchall()
                for row in rows:
                    cid = str(row.id)
                    pid = str(row.parent_id)
                    parent_children.setdefault(pid, []).append(cid)
                    child_ordinals[cid] = int(row.ordinal)

            all_chunk_ids = set(original_chunk_ids)
            for child_ids in parent_children.values():
                all_chunk_ids.update(child_ids)

            span_rows = []
            if all_chunk_ids:
                span_rows = conn.execute(
                    select(
                        chunk_span_table.c.chunk_id,
                        chunk_span_table.c.unit_id,
                        chunk_span_table.c.ordinal,
                        chunk_span_table.c.location,
                        chunk_span_table.c.start_cp,
                        chunk_span_table.c.end_cp,
                    ).where(
                        and_(
                            chunk_span_table.c.tenant_id == tenant_id,
                            chunk_span_table.c.chunk_id.in_(list(all_chunk_ids)),
                        )
                    )
                ).fetchall()

            unit_ids = list({str(row.unit_id) for row in span_rows})
            unit_rows = []
            if unit_ids:
                unit_rows = conn.execute(
                    select(
                        source_unit_table.c.id,
                        source_unit_table.c.text_content,
                        source_unit_table.c.location,
                    ).where(
                        and_(
                            source_unit_table.c.tenant_id == tenant_id,
                            source_unit_table.c.id.in_(unit_ids),
                        )
                    )
                ).fetchall()

        unit_map = {
            str(row.id): {
                "text": str(row.text_content or ""),
                "location": row.location or {},
            }
            for row in unit_rows
        }
        spans_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        for row in span_rows:
            spans_by_chunk.setdefault(str(row.chunk_id), []).append(
                {
                    "unitId": str(row.unit_id),
                    "spanOrdinal": int(row.ordinal),
                    "location": row.location or {},
                    "start": int(row.start_cp),
                    "end": int(row.end_cp),
                }
            )
        for values in spans_by_chunk.values():
            values.sort(key=lambda value: value["spanOrdinal"])

        output = []
        for hit in hits:
            item = dict(hit)
            parent_id = str(hit.get("parentChunkId") or "")
            if parent_id:
                provenance_chunks = parent_children.get(parent_id, [])
            else:
                provenance_chunks = [str(hit.get("chunkId"))] if hit.get("chunkId") else []

            provenance_chunks = sorted(
                provenance_chunks,
                key=lambda cid: child_ordinals.get(cid, 0),
            )
            source_units: List[Dict[str, Any]] = []
            seen_units = set()
            context_text = str(item.get("text") or "")
            cursor = 0

            for cid in provenance_chunks:
                for span in spans_by_chunk.get(cid, []):
                    unit_id = span["unitId"]
                    if unit_id in seen_units:
                        continue
                    seen_units.add(unit_id)
                    unit = unit_map.get(unit_id, {"text": "", "location": {}})
                    unit_text = unit["text"]
                    pos = context_text.find(unit_text, cursor) if unit_text else -1
                    if pos < 0 and unit_text:
                        pos = context_text.find(unit_text)
                    source = {
                        "unitId": unit_id,
                        "location": span["location"] or unit["location"],
                        "start": span["start"],
                        "end": span["end"],
                    }
                    if pos >= 0:
                        source["contextStart"] = pos
                        source["contextEnd"] = pos + len(unit_text)
                        source["contextMapped"] = True
                        cursor = source["contextEnd"]
                    else:
                        # 即使 source unit 经归一化、摘要化或 parent context 被截断，也必须保留 provenance，
                        # 因为回答引用仍需要能追溯到原始文档；此时可以缺失精确 parent offset，
                        # 但 source unit、文档位置和 build 链路必须继续存在，
                        # 不能为了展示处理后的文本而丢掉证据来源链。
                        source["contextMapped"] = False
                    source_units.append(source)

            item["sourceUnits"] = source_units
            output.append(item)
        return output
