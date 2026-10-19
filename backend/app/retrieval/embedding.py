"""Embedding 调用封装。

生产环境通过 LiteLLM Embedding API 获取向量；测试环境没有模型服务时使用确定性向量。
每个批次调用都遵循“先预算预留 → 再校验 run/job 仍有效 → 再发送请求 → 最后按 usage 结算”。
"""
import hashlib
import math
import os
import logging
import time
import uuid
from typing import Iterable, List

from app.config import settings, is_test_env
from app.errors import ApiException
from app.qa.model_clients import get_http_client
from app.model_budget import BudgetService

logger = logging.getLogger(__name__)


def generate_deterministic_embedding(text: str, dim: int = 1024) -> List[float]:
    """仅测试使用：根据文本生成确定性的单位向量，保证测试可重复且不产生真实模型费用。"""
    vec = []
    base_bytes = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(base_bytes + str(i).encode("ascii")).digest()
        val = (int.from_bytes(h[:4], "big") / 2147483648.0) - 1.0
        vec.append(val)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _batched(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _embedding_endpoint() -> str:
    gateway_url = (os.getenv("LITELLM_BASE_URL") or getattr(settings, "LITELLM_BASE_URL", "")).rstrip("/")
    if not gateway_url:
        return ""
    return f"{gateway_url}/embeddings" if gateway_url.endswith("/v1") else f"{gateway_url}/v1/embeddings"


def get_embeddings(
    texts: List[str],
    dim: int = 1024,
    batch_size: int | None = None,
    tenant_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
) -> List[List[float]]:
    if not texts:
        return []

    endpoint = _embedding_endpoint()
    # 生产环境没有 embedding 服务必须失败；确定性向量绝不能作为生产兜底。
    if not endpoint:
        if not is_test_env():
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", "未配置大模型网关地址 LITELLM_BASE_URL，嵌入服务不可用")
        return [generate_deterministic_embedding(t, dim) for t in texts]

    master_key = os.getenv("LITELLM_MASTER_KEY", "")
    batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    # 限制单批大小，避免一次请求体过大或供应商侧超限。
    batch_size = max(1, min(512, batch_size))
    vectors: List[List[float]] = []

    try:
        client = get_http_client()
        for batch in _batched(texts, batch_size):
            call_id = str(uuid.uuid4())
            sent = False
            recorded = False
            call_started = time.perf_counter()
            log_fields = {"call_id": call_id, "purpose": "EMBEDDING", "model_alias": "embedding-main",
                          "tenant_id": tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1"),
                          **({"run_id": run_id} if run_id else {}), **({"job_id": job_id} if job_id else {}),
                          "batch_count": len(batch)}
            logger.info("model_call_started", extra={"fields": log_fields})
            input_tokens = max(1, sum(max(1, len(value) // 4) for value in batch))
            # 在真正发送请求前按估算 Token 预留预算，避免并发调用一起穿透余额。
            BudgetService.reserve(
                tenant_id=tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1"),
                call_id=call_id,
                model_alias="embedding-main",
                input_upper_bound=input_tokens,
                output_upper_bound=0,
                run_id=run_id,
                job_id=job_id,
            )
            try:
                from app.qa.model_gateway import ModelGateway
                effective_tenant = tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1")
                # 预算预留后、真正发请求前再次检查 run 来源和 job 状态；
                # 如果此时撤权/取消，预留会以 NOT_SENT 结算，不产生实际模型费用。
                ModelGateway.assert_run_sources_current(effective_tenant, run_id)
                ModelGateway.assert_job_current(effective_tenant, job_id)
                sent = True
                resp = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {master_key}"},
                    json={"model": "embedding-main", "input": batch, "dimensions": dim},
                    timeout=30.0,
                )
                if resp.status_code != 200:
                    raise ApiException(
                        503,
                        "MODEL_GATEWAY_UNAVAILABLE",
                        f"大模型嵌入接口返回异常状态码 {resp.status_code}: {resp.text}",
                    )
                data = resp.json().get("data", [])
            except Exception:
                from app.qa.model_gateway import ModelGateway
                ModelGateway._record_call(
                    tenant_id=tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1"),
                    call_id=call_id,
                    run_id=run_id,
                    purpose="EMBEDDING",
                    status="FAILED" if sent else "NOT_SENT",
                    input_tokens=0,
                    output_tokens=0,
                    model_alias="embedding-main",
                    job_id=job_id,
                )
                recorded = True
                raise
            # 输入多少条必须返回多少条；不能悄悄接受丢向量的响应。
            if not isinstance(data, list) or len(data) != len(batch):
                raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 返回数量与输入批次不一致")

            rows = []
            seen_indexes = set()
            for item in data:
                if not isinstance(item, dict):
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 返回项格式无效")
                index = item.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(batch) or index in seen_indexes:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 返回 index 缺失、越界或重复")
                seen_indexes.add(index)
                rows.append((index, item.get("embedding")))

            # 供应商返回顺序不一定等于输入顺序，必须按 index 还原后再拼回批次。
            rows.sort(key=lambda item: item[0])
            batch_vectors = []
            for expected_index, (actual_index, vector) in enumerate(rows):
                if actual_index != expected_index:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 返回 index 不连续")
                # 向量维度是索引契约的一部分，错维度不能写入 Milvus。
                if not isinstance(vector, list) or len(vector) != dim:
                    actual = len(vector) if isinstance(vector, list) else 0
                    raise ApiException(
                        502,
                        "INVALID_MODEL_OUTPUT",
                        f"模型生成的嵌入向量维度不符 (预期 {dim}, 得到 {actual})",
                    )
                try:
                    numeric = [float(value) for value in vector]
                except (TypeError, ValueError) as exc:
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 包含非数值元素") from exc
                # NaN/Inf 会污染距离计算；全零向量也没有检索意义。
                if not all(math.isfinite(value) for value in numeric) or not any(value != 0.0 for value in numeric):
                    raise ApiException(502, "INVALID_MODEL_OUTPUT", "Embedding 包含 NaN/Inf 或零向量")
                batch_vectors.append(numeric)

            vectors.extend(batch_vectors)
            usage = resp.json().get("usage") or {}
            reported_tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
            # 供应商没有返回可靠 usage 时，调用虽然成功，但账务状态保留 UNKNOWN，而不是伪造精确费用。
            has_usage = type(reported_tokens) is int and reported_tokens >= 0
            from app.qa.model_gateway import ModelGateway
            ModelGateway._record_call(
                tenant_id=tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1"),
                call_id=call_id,
                run_id=run_id,
                purpose="EMBEDDING",
                status="SUCCESS" if has_usage else "UNKNOWN",
                input_tokens=reported_tokens if has_usage else input_tokens,
                output_tokens=0,
                model_alias="embedding-main",
                job_id=job_id,
            )
            recorded = True

            logger.info("model_call_finished", extra={"fields": {
                **log_fields, "status": "SUCCESS" if has_usage else "UNKNOWN",
                "input_tokens": reported_tokens if has_usage else input_tokens, "output_tokens": 0,
                "usage_reported": has_usage, "duration_ms": round((time.perf_counter() - call_started) * 1000, 2),
            }})

        return vectors
    except ApiException as exc:
        if "log_fields" in locals():
            logger.exception("model_call_failed", extra={"fields": {
                **log_fields, "error_code": exc.code,
                "duration_ms": round((time.perf_counter() - call_started) * 1000, 2),
            }})
        if "call_id" in locals() and "recorded" in locals() and not recorded:
            try:
                from app.qa.model_gateway import ModelGateway
                ModelGateway._record_call(
                    tenant_id=tenant_id or os.getenv("RAG_DEFAULT_TENANT_ID", "tenant-demo-1"),
                    call_id=call_id,
                    run_id=run_id,
                    purpose="EMBEDDING",
                    status="FAILED" if ("sent" in locals() and sent) else "NOT_SENT",
                    input_tokens=0,
                    output_tokens=0,
                    model_alias="embedding-main",
                    job_id=job_id,
                )
            except Exception:
                logger.exception("Failed to reconcile embedding budget after validation error")
        raise
    except Exception as exc:
        logger.exception("model_call_failed", extra={"fields": {
            **(log_fields if "log_fields" in locals() else {}), "error_code": type(exc).__name__,
            **({"duration_ms": round((time.perf_counter() - call_started) * 1000, 2)} if "call_started" in locals() else {}),
        }})
        if not is_test_env():
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", f"大模型嵌入网关连接失败: {exc}") from exc
        return [generate_deterministic_embedding(t, dim) for t in texts]
