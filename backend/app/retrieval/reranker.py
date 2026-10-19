"""Reranker 适配层。

把 ModelGateway.rerank 的 index/score 结果映射回原始 hit。
fail_open 只允许对“临时模型服务/格式故障”退回原排名；
权限、预算、取消等业务错误不在 fallback_codes 中，必须继续向上抛出。
"""
from typing import Any, Dict, List, Optional, Tuple

from app.errors import ApiException
from app.qa.model_gateway import ModelGateway


def rerank_hits(
    tenant_id: str,
    run_id: Optional[str],
    query: str,
    hits: List[Dict[str, Any]],
    model_alias: str,
    top_k: int,
    fail_open: bool,
    job_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    # 空候选不产生模型调用。
    if not hits:
        return [], []

    try:
        ranked = ModelGateway.rerank(
            tenant_id=tenant_id,
            run_id=run_id,
            query=query,
            documents=[str(h.get("text") or "") for h in hits],
            top_n=min(top_k, len(hits)),
            model_alias=model_alias,
            job_id=job_id,
        )
    except ApiException as exc:
        fallback_codes = {
            "RERANK_UNAVAILABLE",
            "MODEL_GATEWAY_UNAVAILABLE",
            "MODEL_TIMEOUT",
            "MODEL_PROVIDER_ERROR",
            "MODEL_RETRYABLE",
            "INVALID_MODEL_OUTPUT",
        }
        # fail-open 不能成为绕过权限/预算的通道；只有列出的供应商/格式类错误允许退回原排序。
        if not fail_open or exc.code not in fallback_codes:
            raise
        return hits[:top_k], [f"RERANK_FALLBACK:{exc.code}"]

    # Reranker 只返回原候选 index，正文仍从原 hits 取，避免信任模型返回的文档内容。
    output: List[Dict[str, Any]] = []
    seen = set()
    for item in ranked:
        idx = item.get("index")
        # 越界/重复 index 直接忽略，防止一个候选重复占据多个排名位置。
        if not isinstance(idx, int) or idx < 0 or idx >= len(hits) or idx in seen:
            continue
        seen.add(idx)
        hit = dict(hits[idx])
        hit["rerankScore"] = float(item.get("relevance_score", item.get("score", 0.0)) or 0.0)
        output.append(hit)

    # 完全无有效结果时，仅在显式 fail_open 下回退；否则把无效模型输出暴露为错误。
    if not output:
        if fail_open:
            return hits[:top_k], ["RERANK_FALLBACK:INVALID_MODEL_OUTPUT"]
        raise ApiException(502, "INVALID_MODEL_OUTPUT", "Reranker 未返回有效排序结果")

    return output[:top_k], []
