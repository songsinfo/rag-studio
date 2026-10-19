# 检索工具图

统一 Agent 使用 `RagRetrievalGraph` 执行权限过滤、查询扩展、多路召回、融合、重排与证据回填。
图定义在 `backend/app/retrieval/rag_graph.py`。
在 backend 目录执行 `python scripts/export_rag_retrieval_graph.py --write` 更新以下图示。

<!-- compiled-rag-retrieval-graph:begin -->
```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	resolve_scope(resolve_scope)
	no_scope(no_scope)
	prepare_query(prepare_query)
	dense_recall(dense_recall)
	bm25_recall(bm25_recall)
	rrf_fusion(rrf_fusion)
	hydrate_and_acl(hydrate_and_acl)
	rerank(rerank)
	mmr(mmr)
	parent_expansion(parent_expansion)
	context_budget(context_budget)
	finalize_hits(finalize_hits)
	__end__([<p>__end__</p>]):::last
	__start__ --> resolve_scope;
	bm25_recall --> rrf_fusion;
	context_budget --> finalize_hits;
	dense_recall --> rrf_fusion;
	hydrate_and_acl --> rerank;
	mmr --> parent_expansion;
	parent_expansion --> context_budget;
	prepare_query --> bm25_recall;
	prepare_query --> dense_recall;
	rerank --> mmr;
	resolve_scope -.-> no_scope;
	resolve_scope -.-> prepare_query;
	rrf_fusion --> hydrate_and_acl;
	finalize_hits --> __end__;
	no_scope --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
<!-- compiled-rag-retrieval-graph:end -->
