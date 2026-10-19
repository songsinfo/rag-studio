"""检索配置的解析、边界约束与发布态加载。

这里的目标不是“尽量接受任意配置”，而是把 API/数据库中的松散 JSON
收敛成一个稳定的 RetrievalConfig。后续检索流水线只依赖这个强约束对象，
避免每个阶段重复判断缺省值、非法范围和不支持的策略组合。
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import and_, select

from app.db import get_db_connection
from app.errors import ApiException
from app.models import config_revision_table


DEFAULT_RETRIEVAL_CONFIG_ID = "cfg-enterprise-retrieval-v2"
VALID_RETRIEVAL_MODES = {"DENSE", "BM25", "HYBRID"}


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """把整数配置限制在安全范围内，异常输入退回默认值。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any, default: bool) -> bool:
    """兼容 bool / 字符串 / 数字形式的开关配置。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class RetrievalConfig:
    """运行时使用的强类型检索配置。

    API payload 可以缺字段，但进入 Pipeline 前必须已经补齐默认值并通过组合校验。
    """
    config_id: str
    mode: str = "HYBRID"
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fusion_top_k: int = 20
    rrf_enabled: bool = True
    rrf_k: int = 60
    final_top_k: int = 8
    rerank_enabled: bool = False
    rerank_top_k: int = 10
    rerank_model: str = "rerank-main"
    rerank_fail_open: bool = True
    parent_expansion_enabled: bool = True
    parent_max_chars: int = 6000
    rewrite_enabled: bool = False
    allow_rewrite_fallback: bool = True
    multi_query_enabled: bool = False
    hyde_enabled: bool = False
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    representations: Tuple[str, ...] = ("ORIGINAL",)
    max_variants: int = 4
    num_candidates: int = 200
    context_max_bytes: int = 6000

    @classmethod
    def from_payload(cls, config_id: str, payload: Optional[Dict[str, Any]]) -> "RetrievalConfig":
        """把外部 JSON 配置转换成稳定的运行时对象。"""
        root = payload if isinstance(payload, dict) else {}
        retrieval = root.get("retrieval") if isinstance(root.get("retrieval"), dict) else root
        rerank = root.get("rerank") if isinstance(root.get("rerank"), dict) else {}
        parent = root.get("parentExpansion") if isinstance(root.get("parentExpansion"), dict) else {}
        query_cfg = root.get("query") if isinstance(root.get("query"), dict) else {}
        post = root.get("post") if isinstance(root.get("post"), dict) else {}
        fusion = root.get("fusion") if isinstance(root.get("fusion"), dict) else {}
        context = root.get("context") if isinstance(root.get("context"), dict) else {}

        raw_mode = str(retrieval.get("mode") or "HYBRID").strip().upper()
        mode = "BM25" if raw_mode in {"SPARSE", "LEXICAL"} else raw_mode

        legacy_top_k = _bounded_int(retrieval.get("topK"), 10, 1, 100)
        per_route_k = _bounded_int(retrieval.get("perRouteK"), max(legacy_top_k * 2, 10), 1, 100)
        dense_default = 0 if mode == "BM25" else per_route_k
        sparse_default = 0 if mode == "DENSE" else per_route_k
        dense_top_k = _bounded_int(retrieval.get("denseTopK"), dense_default, 0, 200)
        sparse_top_k = _bounded_int(retrieval.get("sparseTopK"), sparse_default, 0, 200)
        fusion_top_k = _bounded_int(retrieval.get("fusionTopK"), max(legacy_top_k, min(per_route_k, 30)), 1, 100)
        final_default = _bounded_int(context.get("maxUnits"), min(legacy_top_k, 8), 1, 20)
        final_top_k = _bounded_int(retrieval.get("finalTopK"), final_default, 1, 50)
        num_candidates = _bounded_int(retrieval.get("numCandidates"), max(per_route_k, 200), per_route_k, 1000)
        max_variants = _bounded_int(query_cfg.get("maxVariants"), 4, 1, 4)
        context_max_bytes = _bounded_int(context.get("maxTokens"), 6000, 100, 12000)

        rrf_block = retrieval.get("rrf") if isinstance(retrieval.get("rrf"), dict) else {}
        if not rrf_block and isinstance(root.get("rrf"), dict):
            rrf_block = root.get("rrf")
        rrf_enabled = _as_bool(
            rrf_block.get("enabled", retrieval.get("rrfEnabled")),
            mode == "HYBRID",
        )
        rrf_k = _bounded_int(
            rrf_block.get("k", fusion.get("rrfK", retrieval.get("rrfK"))),
            60,
            1,
            500,
        )

        raw_representations = retrieval.get("representations", root.get("representations", ["ORIGINAL"]))
        if not isinstance(raw_representations, list):
            raw_representations = ["ORIGINAL"]
        representations = tuple(
            str(item).upper()
            for item in raw_representations
            if str(item).upper() in {"ORIGINAL", "SUMMARY", "QUESTION"}
        ) or ("ORIGINAL",)

        try:
            mmr_lambda = float(post.get("lambda", post.get("mmrLambda", 0.7)))
        except (TypeError, ValueError):
            mmr_lambda = 0.7

        return cls(
            config_id=config_id,
            mode=mode,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            fusion_top_k=fusion_top_k,
            rrf_enabled=rrf_enabled,
            rrf_k=rrf_k,
            final_top_k=final_top_k,
            rerank_enabled=_as_bool(rerank.get("enabled", post.get("rerank")), False),
            rerank_top_k=_bounded_int(
                rerank.get("topK", post.get("rerankTopN")),
                min(fusion_top_k, 10),
                1,
                50,
            ),
            rerank_model=str(rerank.get("model") or "rerank-main"),
            rerank_fail_open=_as_bool(
                rerank.get("failOpen", post.get("allowRerankFallback")),
                True,
            ),
            parent_expansion_enabled=_as_bool(parent.get("enabled", post.get("parent")), True),
            parent_max_chars=_bounded_int(parent.get("maxChars"), context_max_bytes, 500, 20000),
            rewrite_enabled=_as_bool(query_cfg.get("rewrite"), False),
            allow_rewrite_fallback=_as_bool(query_cfg.get("allowRewriteFallback"), True),
            multi_query_enabled=_as_bool(query_cfg.get("multiQuery"), False),
            hyde_enabled=_as_bool(query_cfg.get("hyde"), False),
            mmr_enabled=_as_bool(post.get("mmr"), False),
            mmr_lambda=mmr_lambda,
            representations=representations,
            max_variants=max_variants,
            num_candidates=num_candidates,
            context_max_bytes=context_max_bytes,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_config(config: RetrievalConfig, status_code: int = 400) -> RetrievalConfig:
    """校验“单字段合法”之外的策略组合约束。"""
    if config.mode not in VALID_RETRIEVAL_MODES:
        raise ApiException(status_code, "INVALID_CONFIG", f"不支持的检索模式: {config.mode}")
    if config.mode == "DENSE" and config.dense_top_k <= 0:
        raise ApiException(status_code, "INVALID_CONFIG", "DENSE 模式 denseTopK 必须大于 0")
    if config.mode == "BM25" and config.sparse_top_k <= 0:
        raise ApiException(status_code, "INVALID_CONFIG", "BM25 模式 sparseTopK 必须大于 0")
    if config.mode == "HYBRID":
        if config.dense_top_k <= 0 or config.sparse_top_k <= 0:
            raise ApiException(status_code, "INVALID_CONFIG", "HYBRID 模式需要 Dense 与 BM25 两路召回")
        if not config.rrf_enabled:
            raise ApiException(status_code, "INVALID_CONFIG", "HYBRID 模式当前必须启用 RRF 融合")
    if config.final_top_k > config.fusion_top_k:
        raise ApiException(status_code, "INVALID_CONFIG", "finalTopK 不能大于 fusionTopK")
    if config.rerank_enabled and config.rerank_top_k > config.fusion_top_k:
        raise ApiException(status_code, "INVALID_CONFIG", "rerank.topK 不能大于 fusionTopK")
    if config.hyde_enabled and config.mode not in {"DENSE", "HYBRID"}:
        raise ApiException(status_code, "INVALID_CONFIG", "HyDE 需要 Dense 通道")
    if not 0.0 <= config.mmr_lambda <= 1.0:
        raise ApiException(status_code, "INVALID_CONFIG", "MMR lambda 必须在 0 到 1 之间")
    if not config.representations or not set(config.representations).issubset({"ORIGINAL", "SUMMARY", "QUESTION"}):
        raise ApiException(status_code, "INVALID_CONFIG", "representations 配置无效")
    if config.num_candidates < max(config.dense_top_k, config.sparse_top_k):
        raise ApiException(status_code, "INVALID_CONFIG", "numCandidates 不能小于单路召回数量")
    return config


def load_retrieval_config(tenant_id: str, config_id: Optional[str]) -> RetrievalConfig:
    """从已发布配置中加载检索策略；没有显式 ID 时读取默认配置。"""
    effective_id = config_id or DEFAULT_RETRIEVAL_CONFIG_ID
    # 只加载当前租户自己的配置，不能跨租户复用配置 ID。
    with get_db_connection() as conn:
        row = conn.execute(
            select(
                config_revision_table.c.kind,
                config_revision_table.c.status,
                config_revision_table.c.payload,
            ).where(
                and_(
                    config_revision_table.c.tenant_id == tenant_id,
                    config_revision_table.c.id == effective_id,
                )
            )
        ).first()

    if not row:
        if effective_id in {DEFAULT_RETRIEVAL_CONFIG_ID, "cfg-default-retrieval"}:
            return _validate_config(RetrievalConfig(config_id=effective_id), status_code=409)
        raise ApiException(409, "RETRIEVAL_CONFIG_NOT_FOUND", "检索配置不存在或已失效")

    if str(row.kind).upper() != "RETRIEVAL":
        raise ApiException(409, "INVALID_RETRIEVAL_CONFIG", "配置类型不是 RETRIEVAL")
    if str(row.status).upper() not in {"ACTIVE", "PUBLISHED"}:
        raise ApiException(409, "RETRIEVAL_CONFIG_NOT_PUBLISHED", "检索配置尚未发布")

    try:
        return _validate_config(RetrievalConfig.from_payload(effective_id, row.payload), status_code=409)
    except ApiException as exc:
        raise ApiException(409, "INVALID_RETRIEVAL_CONFIG", exc.message)


def validate_retrieval_payload(payload: Dict[str, Any]) -> RetrievalConfig:
    """仅校验一段候选 payload，不访问数据库；评估/配置保存前可复用。"""
    if not isinstance(payload, dict):
        raise ApiException(400, "INVALID_CONFIG", "检索配置必须是 JSON 对象")
    return _validate_config(RetrievalConfig.from_payload("validation", payload))
