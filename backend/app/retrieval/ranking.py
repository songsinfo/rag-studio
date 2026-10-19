from collections import defaultdict
from typing import List, Dict, Any


def rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    weights: List[float] = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """使用 Reciprocal Rank Fusion 合并多路排名。

    RRF 不直接比较 BM25 分数和向量分数，而只利用各 route 中的相对名次：
    score = Σ weight / (rrf_k + rank)。
    这让不同检索器的分数量纲不一致时仍能稳定融合。
    """
    if not ranked_lists:
        return []

    # 未显式给权重时，每条 route 等权。
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = defaultdict(float)
    items_by_id: Dict[str, Dict[str, Any]] = {}

    for route_idx, route_hits in enumerate(ranked_lists):
        w = weights[route_idx]
        # 同一路由里重复出现同一 id 只按第一次排名计分，避免重复候选刷高分。
        seen = set()
        unique_rank = 0
        for hit in route_hits:
            hit_id = hit["id"]
            if hit_id in seen:
                continue
            seen.add(hit_id)
            unique_rank += 1
            if hit_id not in items_by_id:
                items_by_id[hit_id] = dict(hit)
            # rrf_k 越大，头部名次之间的差异越平缓；默认 60 是常用稳健值。
            scores[hit_id] += w / (rrf_k + unique_rank)

    # 分数相同时用 id 做稳定排序，保证重复实验结果可复现。
    sorted_ids = sorted(scores.keys(), key=lambda x: (-scores[x], str(x)))
    fused: List[Dict[str, Any]] = []
    for sid in sorted_ids:
        item = items_by_id[sid]
        item["score"] = scores[sid]
        fused.append(item)

    return fused
