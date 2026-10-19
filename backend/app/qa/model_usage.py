"""Account for every provider attempt using the shared budget service."""
import json
import logging
import time
import uuid

from app.errors import ApiException
from app.model_budget import BudgetService
from app.qa.runtime import AgentRuntimePolicy, get_run_usage
from app.telemetry import model_observation, finish_model_observation, content_enabled, model_response_content, set_model_io

logger = logging.getLogger(__name__)


def token_ceiling(payload):
    # UTF-8 bytes 上界比“每 3 字符一个 token”更保守，包含工具 schema。
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 256


def normalize_provider_usage(usage):
    """将 LangChain/LiteLLM 的用量形状归一化，保留 0 与未知的区别。"""
    source = usage if isinstance(usage, dict) else {}
    inner = source.get("input_token_details")
    inner = inner if isinstance(inner, dict) else {}
    # OpenAI-compatible API 也可能把 cached_tokens 放在 prompt_tokens_details。
    details = source.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    candidates = [
        source.get("cache_read_tokens"), source.get("cached_tokens"),
        inner.get("cache_read"), inner.get("cached_tokens"), details.get("cached_tokens"),
    ]
    cache_read = next((n for n in candidates if type(n) is int and n >= 0), None)
    normalized = dict(source)
    normalized["cache_read_tokens"] = cache_read
    return normalized


def invoke_tracked(fn, *, tenant_id, run_id=None, job_id=None, purpose, alias,
                   payload, max_output_tokens=0, usage_reader=None):
    from app.qa.model_gateway import ModelGateway

    input_limit = token_ceiling(payload)
    output_limit = max(0, int(max_output_tokens))
    if run_id:
        policy = AgentRuntimePolicy.from_env()
        usage = get_run_usage(tenant_id, run_id)
        if usage["modelCalls"] >= policy.max_model_calls or usage["totalTokens"] + input_limit + output_limit > policy.max_total_tokens:
            raise ApiException(429, "BUDGET_EXCEEDED", "本次模型调用将超过调用或 Token 预算")
    call_id = str(uuid.uuid4())
    BudgetService.reserve(tenant_id, call_id, alias, input_limit, output_limit,
                          run_id=run_id, job_id=job_id)
    sent = False
    started = time.perf_counter()
    observation = None
    fields = {"tenant_id": tenant_id, "run_id": run_id, "job_id": job_id,
              "call_id": call_id, "purpose": purpose, "model_alias": alias}
    # Never include payloads: they contain questions, retrieved documents and prompts.
    logger.info("model_call_started", extra={"fields": {k: v for k, v in fields.items() if v is not None}})
    try:
        ModelGateway.assert_run_sources_current(tenant_id, run_id, allow_eval_judge=purpose == "JUDGE")
        ModelGateway.assert_job_current(tenant_id, job_id)
        from app.evaluation.diagnostics import capture_model_context
        if capture_model_context(tenant_id, run_id, call_id, purpose, payload):
            # Private artifact I/O may take time; revalidate at the send boundary.
            ModelGateway.assert_run_sources_current(tenant_id, run_id, allow_eval_judge=purpose == "JUDGE")
            ModelGateway.assert_job_current(tenant_id, job_id)
        sent = True
        # One observation per actual attempt; retry/fallback invokes this function again.
        # Errors from the model function itself must pass through unchanged.
        # Exporter faults are caught inside the optional observation adapter.
        with model_observation(purpose, alias, call_id) as active:
            observation = active
            set_model_io(active, input=payload)
            result = fn()
            if active is not None and content_enabled("MODEL"):
                try:
                    ModelGateway.assert_run_sources_current(tenant_id, run_id, allow_eval_judge=purpose == "JUDGE")
                    ModelGateway.assert_job_current(tenant_id, job_id)
                    set_model_io(active, output=model_response_content(result))
                except Exception:
                    logger.warning("model_content_capture_skipped")
        if getattr(result, "status_code", 200) >= 400:
            raise ApiException(503, "MODEL_GATEWAY_UNAVAILABLE", f"模型网关返回状态码 {result.status_code}")
        usage = normalize_provider_usage(usage_reader(result) if usage_reader else {})
        reported = bool(usage and type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int
                        and usage["input_tokens"] >= 0 and usage["output_tokens"] >= 0)
        detail = {
            "cacheReadTokens": usage["cache_read_tokens"],
            "inputTokensReported": usage.get("input_tokens") if reported else None,
            "model": alias, "durationMs": round((time.perf_counter() - started) * 1000, 2),
            "usageEstimator": "provider" if reported else "byte-ceiling",
        }
        # 不把缺失的缓存字段当作 0；已返回且合法的 0 才是真实零命中。
        if not isinstance(detail["cacheReadTokens"], int) or detail["cacheReadTokens"] < 0:
            detail["cacheReadTokens"] = None
        settlement = ModelGateway._record_call(
            tenant_id, call_id, run_id, purpose, "SUCCESS" if reported else "UNKNOWN",
            usage["input_tokens"] if reported else input_limit,
            usage["output_tokens"] if reported else output_limit, alias, job_id=job_id,
            usage_detail=detail,
        )
        finish_model_observation(observation, status="SUCCESS" if reported else "UNKNOWN",
            duration_ms=(time.perf_counter() - started) * 1000,
            settlement=settlement, usage_source="PROVIDER" if reported else "BYTE_CEILING",
            reported_usage={"input": usage["input_tokens"], "output": usage["output_tokens"],
                            **({"cache_read_input_tokens": usage["cache_read_tokens"]}
                               if usage["cache_read_tokens"] is not None else {})} if reported else None)
        logger.info("model_call_finished", extra={"fields": {
            **{k: v for k, v in fields.items() if v is not None},
            "status": "SUCCESS" if reported else "UNKNOWN",
            "input_tokens": usage["input_tokens"] if reported else input_limit,
            "output_tokens": usage["output_tokens"] if reported else output_limit,
            "usage_reported": reported, "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }})
        return result
    except Exception as exc:
        ModelGateway._record_call(tenant_id, call_id, run_id, purpose,
            "FAILED" if sent else "NOT_SENT", input_limit if sent else 0,
            output_limit if sent else 0, alias, job_id=job_id)
        finish_model_observation(observation, status="FAILED" if sent else "NOT_SENT",
            duration_ms=(time.perf_counter() - started) * 1000,
            error_code=exc.code if isinstance(exc, ApiException) else type(exc).__name__)
        logger.exception("model_call_failed", extra={"fields": {
            **{k: v for k, v in fields.items() if v is not None},
            "status": "FAILED" if sent else "NOT_SENT",
            "error_code": exc.code if isinstance(exc, ApiException) else type(exc).__name__,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }})
        raise
