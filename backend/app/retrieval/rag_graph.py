"""供统一 Agent 的 search_knowledge 及历史 RAG 兼容执行调用的 LangGraph 检索图。

新建在线问答统一进入 DURABLE Agent/QA_RUN Job；Agent 的知识检索工具调用本图。
独立 Search API 与纯检索评测仍走 RetrievalPipeline.retrieve()，共享底层 Milvus、
RRF、Reranker、Parent Expansion 和 provenance。内层检索图不启用 Checkpointer；
单次问答的持久化恢复由外层 Agent/QA_RUN 负责，不能恢复本图任意中间节点。
"""
from __future__ import annotations

import logging
import threading
import functools
import itertools
from contextvars import copy_context
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.db import get_db_connection
from app.observability import agent_span
from app.telemetry import content_enabled, export_current_span, set_span_io
from app.errors import ApiException
from app.identity.deps import Actor, assert_hits_still_readable, get_readable_build_records
from app.retrieval.config import RetrievalConfig, load_retrieval_config
from app.retrieval.milvus_service import MilvusHybridService
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.ranking import rrf_fuse
from app.retrieval.reranker import rerank_hits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagRetrievalContext:
    """运行时依赖不放入图状态，避免把 Actor、连接和回调当作检索结果。"""

    actor: Actor
    query: str
    kb_ids: Optional[List[str]]
    document_ids: Optional[List[str]]
    config_id: Optional[str]
    run_id: Optional[str]
    job_id: Optional[str]
    limit_override: Optional[int]
    history: List[Dict[str, str]]
    record_event_fn: Optional[Callable[..., Any]]
    retrieval_index: int = 0
    observation_context: Any = None


class RagRetrievalState(TypedDict, total=False):
    """阶段结果只在本次请求内传递；未启用持久化 checkpoint。"""

    cfg: RetrievalConfig
    document_ids: Optional[List[str]]
    allowed_build_ids: List[str]
    allowed_build_epochs: Dict[str, int]
    independent_query: str
    query_variants: List[str]
    hyde_text: Optional[str]
    warnings: List[str]
    dense_routes: List[List[Dict[str, Any]]]
    sparse_routes: List[List[Dict[str, Any]]]
    dense_weight: float
    sparse_weight: float
    dense_duration_ms: float
    sparse_duration_ms: float
    hits: List[Dict[str, Any]]


class RagRetrievalGraph:
    """固定 DAG：scope → query → Dense/BM25 → RRF → Rerank → Context。"""

    _graph = None
    _graph_lock = threading.Lock()
    _retrieval_indices = itertools.count(1)

    @classmethod
    def _get_graph(cls):
        if cls._graph is not None:
            return cls._graph
        with cls._graph_lock:
            if cls._graph is not None:
                return cls._graph

            builder = StateGraph(RagRetrievalState, context_schema=RagRetrievalContext)
            builder.add_node("resolve_scope", cls._observe_stage("scope", cls._resolve_scope))
            builder.add_node("no_scope", cls._observe_stage("no_scope", cls._no_scope))
            builder.add_node("prepare_query", cls._observe_stage("prepare_query", cls._prepare_query))
            builder.add_node("dense_recall", cls._observe_stage("dense_recall", cls._dense_recall))
            builder.add_node("bm25_recall", cls._observe_stage("bm25_recall", cls._bm25_recall))
            builder.add_node("rrf_fusion", cls._observe_stage("rrf_fusion", cls._rrf_fusion))
            builder.add_node("hydrate_and_acl", cls._observe_stage("hydrate_and_acl", cls._hydrate_and_acl))
            builder.add_node("rerank", cls._observe_stage("rerank", cls._rerank))
            builder.add_node("mmr", cls._observe_stage("mmr", cls._mmr))
            builder.add_node("parent_expansion", cls._observe_stage("parent_expansion", cls._parent_expansion))
            builder.add_node("context_budget", cls._observe_stage("context_budget", cls._context_budget))
            builder.add_node("finalize_hits", cls._observe_stage("finalize_hits", cls._finalize_hits))

            builder.add_edge(START, "resolve_scope")
            builder.add_conditional_edges(
                "resolve_scope",
                lambda s: "prepare_query" if s["allowed_build_ids"] else "no_scope",
                {"prepare_query": "prepare_query", "no_scope": "no_scope"},
            )
            builder.add_edge("no_scope", END)
            builder.add_edge("prepare_query", "dense_recall")
            builder.add_edge("prepare_query", "bm25_recall")
            # 两路检索完成后统一融合：并行节点只写各自的 State 字段。
            builder.add_edge(["dense_recall", "bm25_recall"], "rrf_fusion")
            builder.add_edge("rrf_fusion", "hydrate_and_acl")
            builder.add_edge("hydrate_and_acl", "rerank")
            builder.add_edge("rerank", "mmr")
            builder.add_edge("mmr", "parent_expansion")
            builder.add_edge("parent_expansion", "context_budget")
            builder.add_edge("context_budget", "finalize_hits")
            builder.add_edge("finalize_hits", END)

            cls._graph = builder.compile()
            return cls._graph

    @staticmethod
    def _observe_stage(name, node):
        @functools.wraps(node)
        def observed(state, runtime):
            context = runtime.context.observation_context
            if context is not None:
                return context.copy().run(in_context, state, runtime)
            return in_context(state, runtime)

        def in_context(state, runtime):
            idx = runtime.context.retrieval_index
            with agent_span("rag." + name, {"rag.retrieval_index": idx, "rag.stage": name}) as span:
                # LangGraph's parallel nodes run with their own captured context,
                # rather than reusing a global trace or span across different runs.
                RagRetrievalGraph._capture_io(runtime.context, state, span, "input")
                result = node(state, runtime)
                RagRetrievalGraph._capture_io(runtime.context, result, span, "output")
                if isinstance(result, dict):
                    from app.evaluation.diagnostics import capture_eval_stage
                    from opentelemetry import trace

                    span = trace.get_current_span()
                    cfg = result.get("cfg") or state.get("cfg")
                    hits = result.get("hits")
                    if isinstance(hits, list):
                        capture_eval_stage(
                            runtime.context, name, hits,
                            limit=max(1, getattr(cfg, "fusion_top_k", 50)),
                            context=name in {"context_budget", "finalize_hits"},
                        )
                        if span and span.is_recording():
                            span.set_attribute("rag.hit_count", len(hits))

                    # Dense/BM25 nodes produce per-query ranked routes, NOT hits.
                    # Capture before RRF so Gold dropped at fusion stays attributable.
                    route_field = {
                        "dense_recall": ("dense_routes", "dense_top_k"),
                        "bm25_recall": ("sparse_routes", "sparse_top_k"),
                    }.get(name)
                    if route_field:
                        routes = result.get(route_field[0]) or []
                        count = 0
                        for route_index, route_hits in enumerate(routes):
                            if not isinstance(route_hits, list):
                                continue
                            count += len(route_hits)
                            capture_eval_stage(
                                runtime.context, name, route_hits,
                                limit=max(1, getattr(cfg, route_field[1], 50)),
                                route_index=route_index,
                            )
                        if span and span.is_recording():
                            span.set_attribute("rag.hit_count", count)
                            span.set_attribute("rag.route_count", len(routes))
                return result
        return observed

    @staticmethod
    def _capture_io(ctx, data, span, field):
        if not content_enabled("RETRIEVAL") or not export_current_span():
            return
        try:
            from app.qa.model_gateway import ModelGateway
            ModelGateway.assert_run_sources_current(ctx.actor.tenant_id, ctx.run_id)
            ModelGateway.assert_job_current(ctx.actor.tenant_id, ctx.job_id)
            hits = list(data.get("hits") or [])
            for key in ("dense_routes", "sparse_routes"):
                hits.extend(hit for route in data.get(key, []) for hit in route)
            with get_db_connection() as conn:
                assert_hits_still_readable(conn, ctx.actor, hits, ctx.kb_ids,
                    data.get("document_ids", ctx.document_ids))
            snapshot = dict(data)
            if snapshot.get("cfg") is not None:
                snapshot["cfg"] = snapshot["cfg"].as_dict()
            if field == "input":
                snapshot = {"query": ctx.query, "history": ctx.history,
                    "kbIds": ctx.kb_ids, "documentIds": ctx.document_ids, "state": snapshot}
            set_span_io("RETRIEVAL", span=span, **{field: snapshot})
        except Exception:
            logger.warning("retrieval_content_capture_skipped")

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
        record_event_fn: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Agent 检索工具与独立检索共用；内部图无需持久化 checkpoint。"""
        ctx = RagRetrievalContext(
            actor=actor,
            query=query,
            kb_ids=kb_ids,
            document_ids=document_ids,
            config_id=config_id,
            run_id=run_id,
            job_id=job_id,
            limit_override=limit_override,
            history=list(history or []),
            record_event_fn=record_event_fn,
            retrieval_index=next(cls._retrieval_indices),
        )
        with agent_span("rag.retrieve", {"rag.retrieval_index": ctx.retrieval_index}) as span:
            ctx = replace(ctx, observation_context=copy_context())
            cls._capture_io(ctx, {}, span, "input")
            state = cls._get_graph().invoke({}, context=ctx)
            result = {
                "hits": state.get("hits", []),
                "warnings": state.get("warnings", []),
                "config": state["cfg"].as_dict(),
                "allowedBuildCount": len(state["allowed_build_ids"]),
            }
            cls._capture_io(ctx, result, span, "output")
            return result

    @staticmethod
    def _record(ctx: RagRetrievalContext, stage: str, elapsed_ms: float, count: int, **fields) -> None:
        """本地阶段事件只落数量、参数和耗时；可选正文由独立观测入口记录。"""
        if ctx.record_event_fn is None or not ctx.run_id:
            return
        try:
            ctx.record_event_fn(
                ctx.actor.tenant_id,
                ctx.run_id,
                stage,
                {"hitCount": count, "durationMs": round(elapsed_ms, 2), "retrievalIndex": ctx.retrieval_index, **fields},
            )
        except Exception:
            # 可观测性故障不能改变检索结果。
            logger.exception("rag_retrieval_stage_event_failed", extra={"fields": {"stage": stage}})

    @classmethod
    def _resolve_scope(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        started = time.perf_counter()
        cfg = load_retrieval_config(ctx.actor.tenant_id, ctx.config_id)
        document_ids = ctx.document_ids
        with get_db_connection() as conn:
            if ctx.run_id:
                from app.qa.context import run_document_scope

                run_scope = run_document_scope(conn, ctx.actor, ctx.run_id)
                if run_scope is not None:
                    document_ids = run_scope if document_ids is None else list(set(document_ids).intersection(run_scope))
            readable = get_readable_build_records(conn, ctx.actor, ctx.kb_ids, document_ids)

        # 与原 RetrievalPipeline 保持一致：索引能力不匹配时失败关闭。
        if cfg.mode in {"DENSE", "HYBRID"} or cfg.mmr_enabled:
            for row in readable:
                dims = row.get("embeddingDims")
                fingerprint = row.get("modelFingerprint")
                if (dims is not None and dims != 1024) or (
                    fingerprint is not None and fingerprint != "embedding-main"
                ):
                    raise ApiException(
                        409, "INDEX_CAPABILITY_MISSING",
                        "当前生效构建的 Embedding 模型或维度与检索配置不兼容，请重新构建索引",
                    )
                available = set((row.get("capabilities") or {}).get("representations") or ["ORIGINAL"])
                if not set(cfg.representations).issubset(available):
                    raise ApiException(
                        409, "INDEX_CAPABILITY_MISSING",
                        "当前生效构建缺少检索所需 representation，请重新构建索引",
                    )

        allowed = [row["buildId"] for row in readable]
        cls._record(ctx, "RAG_SCOPE", (time.perf_counter() - started) * 1000, len(allowed))
        return {
            "cfg": cfg,
            "document_ids": document_ids,
            "allowed_build_ids": allowed,
            "allowed_build_epochs": {
                row["buildId"]: int(row.get("indexEpoch") or 0) for row in readable
            },
            "warnings": [],
            "hits": [],
        }

    @staticmethod
    def _no_scope(state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        # 无可读 build 时不做 Rewrite / Embedding / Milvus / Reranker 调用。
        return {"hits": []}

    @classmethod
    def _prepare_query(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        started = time.perf_counter()
        warnings: List[str] = []
        independent, variants, hyde = RetrievalPipeline._prepare_queries(
            actor=ctx.actor,
            query=ctx.query,
            history=ctx.history,
            cfg=state["cfg"],
            run_id=ctx.run_id,
            job_id=ctx.job_id,
            warnings=warnings,
        )
        cls._record(
            ctx, "RAG_PREPARE_QUERY", (time.perf_counter() - started) * 1000,
            len(variants), hydeEnabled=bool(hyde),
        )
        return {
            "independent_query": independent,
            "query_variants": variants,
            "hyde_text": hyde,
            "warnings": warnings,
        }

    @classmethod
    def _recall_routes(
        cls, state: RagRetrievalState, ctx: RagRetrievalContext, *, dense: bool
    ) -> Dict[str, Any]:
        cfg = state["cfg"]
        if dense:
            enabled = cfg.mode in {"DENSE", "HYBRID"}
            queries = list(state["query_variants"])
            if state.get("hyde_text"):
                queries.append(state["hyde_text"])
        else:
            enabled = cfg.mode in {"BM25", "HYBRID"}
            queries = list(state["query_variants"])
        routes: List[List[Dict[str, Any]]] = []
        started = time.perf_counter()
        if enabled:
            for candidate_query in queries:
                route = MilvusHybridService.search(
                    tenant_id=ctx.actor.tenant_id,
                    query=candidate_query,
                    allowed_build_ids=state["allowed_build_ids"],
                    dense_top_k=cfg.dense_top_k if dense else 0,
                    sparse_top_k=0 if dense else cfg.sparse_top_k,
                    fusion_top_k=cfg.dense_top_k if dense else cfg.sparse_top_k,
                    mode="DENSE" if dense else "BM25",
                    rrf_enabled=False,
                    rrf_k=cfg.rrf_k,
                    allowed_build_epochs=state["allowed_build_epochs"],
                    representations=list(cfg.representations),
                    run_id=ctx.run_id,
                    job_id=ctx.job_id,
                    num_candidates=cfg.num_candidates,
                )
                if route:
                    routes.append(route)
        weight = (
            (0.5 if cfg.mode == "HYBRID" else 1.0) / max(1, len(queries))
            if enabled else 0.0
        )
        prefix = "dense" if dense else "sparse"
        return {
            f"{prefix}_routes": routes,
            f"{prefix}_weight": weight,
            f"{prefix}_duration_ms": (time.perf_counter() - started) * 1000,
        }

    @classmethod
    def _dense_recall(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        return cls._recall_routes(state, runtime.context, dense=True)

    @classmethod
    def _bm25_recall(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        return cls._recall_routes(state, runtime.context, dense=False)

    @classmethod
    def _rrf_fusion(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        cfg = state["cfg"]
        # 保持原有顺序：BM25 各变体路线在前，Dense（含 HyDE）在后。
        routes = list(state["sparse_routes"]) + list(state["dense_routes"])
        weights = (
            [state["sparse_weight"]] * len(state["sparse_routes"])
            + [state["dense_weight"]] * len(state["dense_routes"])
        )
        cls._record(
            ctx, "RAG_BM25_RECALL", state["sparse_duration_ms"],
            sum(len(route) for route in state["sparse_routes"]),
            routeCount=len(state["sparse_routes"]),
        )
        cls._record(
            ctx, "RAG_DENSE_RECALL", state["dense_duration_ms"],
            sum(len(route) for route in state["dense_routes"]),
            routeCount=len(state["dense_routes"]),
        )
        started = time.perf_counter()
        if not routes:
            hits = []
        elif len(routes) == 1:
            hits = routes[0][:cfg.fusion_top_k]
        else:
            hits = rrf_fuse(routes, weights=weights, rrf_k=cfg.rrf_k)[:cfg.fusion_top_k]
        cls._record(ctx, "RAG_RRF_FUSION", (time.perf_counter() - started) * 1000, len(hits), rrfK=cfg.rrf_k)
        return {"hits": hits}

    @classmethod
    def _hydrate_and_acl(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        started = time.perf_counter()
        hits = RetrievalPipeline._hydrate_original_chunks(ctx.actor.tenant_id, state["hits"])
        # Reranker 属于外部模型边界，外发之前必须再查一次实时 ACL。
        with get_db_connection() as conn:
            assert_hits_still_readable(conn, ctx.actor, hits, ctx.kb_ids, state["document_ids"])
        cls._record(ctx, "RAG_HYDRATE", (time.perf_counter() - started) * 1000, len(hits))
        return {"hits": hits}

    @classmethod
    def _rerank(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        cfg = state["cfg"]
        hits = state["hits"]
        warnings = list(state["warnings"])
        started = time.perf_counter()
        if cfg.rerank_enabled and hits:
            hits, rerank_warnings = rerank_hits(
                tenant_id=ctx.actor.tenant_id,
                run_id=ctx.run_id,
                query=state["independent_query"],
                job_id=ctx.job_id,
                hits=hits,
                model_alias=cfg.rerank_model,
                top_k=cfg.rerank_top_k,
                fail_open=cfg.rerank_fail_open,
            )
            warnings.extend(rerank_warnings)
        cls._record(
            ctx, "RAG_RERANK", (time.perf_counter() - started) * 1000,
            len(hits), enabled=cfg.rerank_enabled,
        )
        return {"hits": hits, "warnings": warnings}

    @classmethod
    def _mmr(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        cfg = state["cfg"]
        hits = state["hits"]
        started = time.perf_counter()
        if cfg.mmr_enabled and hits:
            hits = RetrievalPipeline._mmr_select(
                ctx.actor.tenant_id, state["independent_query"], hits,
                cfg.final_top_k, cfg.mmr_lambda, run_id=ctx.run_id, job_id=ctx.job_id,
            )
        cls._record(ctx, "RAG_MMR", (time.perf_counter() - started) * 1000, len(hits), enabled=cfg.mmr_enabled)
        return {"hits": hits}

    @classmethod
    def _parent_expansion(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        cfg = state["cfg"]
        hits = state["hits"]
        warnings = list(state["warnings"])
        started = time.perf_counter()
        if cfg.parent_expansion_enabled and hits:
            hits = RetrievalPipeline._expand_parents(ctx.actor.tenant_id, hits, cfg, warnings)
        cls._record(
            ctx, "RAG_PARENT_EXPANSION", (time.perf_counter() - started) * 1000,
            len(hits), enabled=cfg.parent_expansion_enabled,
        )
        return {"hits": hits, "warnings": warnings}

    @classmethod
    def _context_budget(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        cfg = state["cfg"]
        started = time.perf_counter()
        budgeted: List[Dict[str, Any]] = []
        used_bytes = 0
        warnings = list(state["warnings"])
        for hit in state["hits"]:
            size = len(str(hit.get("text") or "").encode("utf-8"))
            if used_bytes + size > cfg.context_max_bytes:
                if not budgeted:
                    # 沿用原有策略：最佳 Parent 太长时退回对应 Child。
                    child_id = str(hit.get("chunkId") or "")
                    fallback = RetrievalPipeline._hydrate_original_chunks(
                        ctx.actor.tenant_id, [{"chunkId": child_id, **hit}]
                    )[0] if child_id else hit
                    if len(str(fallback.get("text") or "").encode("utf-8")) <= cfg.context_max_bytes:
                        budgeted.append(fallback)
                warnings.append("CONTEXT_TRIMMED")
                break
            budgeted.append(hit)
            used_bytes += size
            if len(budgeted) >= cfg.final_top_k:
                break
        cls._record(
            ctx, "RAG_CONTEXT_BUDGET", (time.perf_counter() - started) * 1000,
            len(budgeted), inputCount=len(state["hits"]), usedBytes=used_bytes,
            maxBytes=cfg.context_max_bytes,
        )
        return {"hits": budgeted, "warnings": warnings}

    @classmethod
    def _finalize_hits(cls, state: RagRetrievalState, runtime: Runtime[RagRetrievalContext]) -> Dict[str, Any]:
        ctx = runtime.context
        started = time.perf_counter()
        hits = RetrievalPipeline._attach_provenance(ctx.actor.tenant_id, state["hits"])
        final_limit = state["cfg"].final_top_k
        if ctx.limit_override is not None:
            final_limit = max(1, min(50, int(ctx.limit_override)))
        hits = hits[:final_limit]
        # 下游 QA 即将发送正文给 LLM，返回前再次执行 ACL + active build 检查。
        with get_db_connection() as conn:
            assert_hits_still_readable(conn, ctx.actor, hits, ctx.kb_ids, state["document_ids"])
        cls._record(ctx, "RAG_FINAL_HITS", (time.perf_counter() - started) * 1000, len(hits))
        return {"hits": hits}
