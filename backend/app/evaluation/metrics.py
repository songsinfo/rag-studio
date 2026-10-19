"""评估指标的纯计算实现。

这里尽量保持“无数据库、无模型调用”的纯函数风格，方便用手算样例验证公式。
特别注意所有分母为 0 的情况都返回 None，而不是 0；0 表示“测了但没命中”，
None 表示“这个指标当前不适用/没有可计算样本”，两者在报告中语义完全不同。
"""
import math
from typing import List, Set, Dict, Any, Optional, Union


class EvaluationMetrics:
    """检索、延迟分位数以及 Judge 校准相关指标。"""

    @staticmethod
    def ratio(numerator: int, denominator: int) -> Optional[float]:
        """安全计算比例；分母为 0 时返回 None，表示“当前不可计算”，而不是误导性的 0 分。"""
        # 分母为 0 时不能返回 0.0，否则会把“没有样本”误解成“表现为 0”。
        return None if denominator == 0 else float(numerator) / denominator

    @staticmethod
    def retrieval(
        groups: List[Set[str]],
        ranked: List[Set[str]],
        k: int = 5,
        ranked_grades: Optional[List[int]] = None,
        ideal_grades: Optional[List[int]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        基于 Gold 事实组计算 Hit@K、Recall@K、MRR@K，并可选计算 NDCG@K。

        每个事实组包含一组等价 anchor 标识；某个排名位置只要与事实组集合存在交集，
        就视为命中该事实组。同一事实组即使被多个候选覆盖，也只计一次召回。
        """
        if k < 1:
            raise ValueError("k must be positive")
        result: Dict[str, Optional[float]] = {}
        # 没有 Gold 事实组时，检索指标“不适用”，不能硬填 0。
        if not groups:
            result["hit"] = None
            result["recall"] = None
            result["mrr"] = None
            if ranked_grades is not None and ideal_grades is not None:
                result["ndcg"] = None
            return result

        if any(len(g) == 0 for g in groups):
            raise ValueError("Empty fact group")

        # found 记录“已经命中的 Gold 事实组索引”，同一事实组被多个候选命中也只算一次。
        found: Set[int] = set()
        first_relevant_rank = 0

        for rank in range(min(k, len(ranked))):
            relevant = False
            # 一个候选可能同时覆盖多个 Gold 事实组，因此这里不能命中一个后就 break。
            for g_idx, g_set in enumerate(groups):
                if not g_set.isdisjoint(ranked[rank]):
                    found.add(g_idx)
                    relevant = True
            if relevant and first_relevant_rank == 0:
                first_relevant_rank = rank + 1

        result["hit"] = 1.0 if found else 0.0
        result["recall"] = float(len(found)) / len(groups)
        result["mrr"] = (1.0 / first_relevant_rank) if first_relevant_rank > 0 else 0.0

        if ranked_grades is not None and ideal_grades is not None:
            result["ndcg"] = EvaluationMetrics.ndcg(ranked_grades, ideal_grades, k)

        return result

    @staticmethod
    def ndcg(ranked: List[int], ideal: List[int], k: int = 5) -> Optional[float]:
        """
        计算分级相关性（0..3）的 NDCG@K。

        公式为 DCG@K / IDCG@K；指数增益 (2^r - 1) 强调高相关结果，
        log2 位置折损体现“越靠前越重要”的排序价值。
        """
        if k < 1:
            raise ValueError("k must be positive")
        if any(g < 0 or g > 3 for g in ranked) or any(g < 0 or g > 3 for g in ideal):
            raise ValueError("Grade must be 0..3; unjudged is not zero")

        # IDCG 使用同一组 relevance grade 的理想降序排列，作为归一化上界。
        sorted_ideal = sorted(ideal, reverse=True)
        denominator = EvaluationMetrics.dcg(sorted_ideal, k)
        if denominator == 0.0:
            return None
        return EvaluationMetrics.dcg(ranked, k) / denominator

    @staticmethod
    def dcg(grades: List[int], k: int) -> float:
        """计算 DCG@K：累计前 K 个结果的分级相关性，并按排名位置进行对数折损。"""
        total = 0.0
        for i in range(min(k, len(grades))):
            # 指数增益强调高相关文档；log2(i+2) 对越靠后的排名施加更大折损。
            total += (2.0 ** grades[i] - 1.0) / math.log2(i + 2)
        return total

    @staticmethod
    def percentile(values: List[Union[int, float]], quantile: float) -> Optional[Union[int, float]]:
        """使用 nearest-rank 向上取整规则计算分位数：index = ceil(N * quantile) - 1；小样本下不会做插值。"""
        if quantile <= 0 or quantile > 1:
            raise ValueError("quantile must be (0,1]")
        if not values:
            return None
        sorted_vals = sorted(values)
        # 采用 nearest-rank，而不是插值分位数；小样本下 P95 往往就是最大值。
        idx = math.ceil(len(sorted_vals) * quantile) - 1
        return sorted_vals[idx]

    @staticmethod
    def categorical_comparison(
        pairs: List[tuple[str, str]],
        labels: List[str],
    ) -> Dict[str, Any]:
        """返回可比较标签对的混淆矩阵与原始一致率；无法映射到给定 labels 的样本会被跳过。"""
        # 行是人工标签（视为校准基准），列是自动 Judge 标签。
        matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
        compared = 0
        agreed = 0
        for automatic, human in pairs:
            if automatic not in labels or human not in labels:
                continue
            matrix[human][automatic] += 1
            compared += 1
            agreed += int(automatic == human)
        return {
            "compared": compared,
            "agreementRate": EvaluationMetrics.ratio(agreed, compared),
            "confusionMatrix": matrix,
        }

    @staticmethod
    def correctness_calibration(pairs: List[tuple[str, str]]) -> Dict[str, Any]:
        """对 CORRECT 做二分类校准；UNDETERMINED 没有明确正负语义，因此不进入 FPR/FNR 分母。"""
        # UNDETERMINED 没有明确正负语义，不进入二分类 FPR/FNR 分母。
        judged = [
            (automatic, human)
            for automatic, human in pairs
            if automatic in {"CORRECT", "PARTIAL", "WRONG"}
            and human in {"CORRECT", "PARTIAL", "WRONG"}
        ]
        human_negative = sum(1 for _, human in judged if human != "CORRECT")
        human_positive = sum(1 for _, human in judged if human == "CORRECT")
        # 正类定义为 CORRECT：
        # FP = 人工认为不是 CORRECT，但自动 Judge 判成 CORRECT。
        false_positive = sum(
            1 for automatic, human in judged if automatic == "CORRECT" and human != "CORRECT"
        )
        # FN = 人工认为是 CORRECT，但自动 Judge 没判成 CORRECT。
        false_negative = sum(
            1 for automatic, human in judged if automatic != "CORRECT" and human == "CORRECT"
        )
        base = EvaluationMetrics.categorical_comparison(
            judged, ["CORRECT", "PARTIAL", "WRONG"]
        )
        base.update({
            "falsePositiveRate": EvaluationMetrics.ratio(false_positive, human_negative),
            "falseNegativeRate": EvaluationMetrics.ratio(false_negative, human_positive),
            "falsePositives": false_positive,
            "falseNegatives": false_negative,
        })
        return base

