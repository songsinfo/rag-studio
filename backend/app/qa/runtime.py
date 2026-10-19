"""Agent / QA 运行时保护。

集中处理整体 deadline、模型调用次数、Token 预算、上下文压缩、重试和熔断。
这些限制都是“硬停止条件”，不能依赖模型自己判断什么时候该结束。
"""
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from sqlalchemy import and_, func, select

from app.db import get_db_connection
from app.errors import ApiException
from app.models import model_call_table
from app.qa.business_context import has_business_context, state_from_messages, state_message


T = TypeVar("T")


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """一次 Agent 运行允许消耗的资源上限与重试策略。"""
    overall_timeout_seconds: int = 120
    max_rounds: int = 6
    max_tool_calls: int = 8
    repeated_call_limit: int = 3
    max_context_tokens: int = 20000
    max_total_tokens: int = 50000
    max_model_calls: int = 12
    call_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.25
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: int = 30
    approval_tools: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls) -> "AgentRuntimePolicy":
        """从环境变量读取策略，并把异常/过小值收敛到安全下限。"""
        approval_tools = frozenset(
            item.strip()
            for item in os.getenv("AGENT_APPROVAL_TOOLS", "").split(",")
            if item.strip()
        )
        return cls(
            overall_timeout_seconds=max(5, int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))),
            max_rounds=max(1, int(os.getenv("AGENT_MAX_ROUNDS", "6"))),
            max_tool_calls=max(1, int(os.getenv("AGENT_MAX_TOOL_CALLS", "8"))),
            repeated_call_limit=max(2, int(os.getenv("AGENT_REPEAT_CALL_LIMIT", "3"))),
            max_context_tokens=max(1000, int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "20000"))),
            max_total_tokens=max(1000, int(os.getenv("AGENT_MAX_TOTAL_TOKENS", "50000"))),
            max_model_calls=max(1, int(os.getenv("AGENT_MAX_MODEL_CALLS", "12"))),
            call_max_attempts=max(1, min(5, int(os.getenv("AGENT_CALL_MAX_ATTEMPTS", "3")))),
            retry_base_delay_seconds=max(0.05, float(os.getenv("AGENT_RETRY_BASE_DELAY_SECONDS", "0.25"))),
            circuit_failure_threshold=max(2, int(os.getenv("AGENT_CIRCUIT_FAILURE_THRESHOLD", "5"))),
            circuit_reset_seconds=max(5, int(os.getenv("AGENT_CIRCUIT_RESET_SECONDS", "30"))),
            approval_tools=approval_tools,
        )


def estimate_tokens(value: Any) -> int:
    """无精确 tokenizer 时的保守估算，仅用于运行时保护，不用于真实计费。"""
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # 中文/英文混合场景下用“约 3 字符/token”做保守近似，宁可早一点触发预算。
    # 对中英文混合文本采用偏保守的 token 近似，宁可高估预算也不要因为低估上下文长度导致模型请求超限。
    return max(1, (len(value) + 2) // 3)


def _to_langchain_message(message: Dict[str, Any]):
    role = str(message.get("role") or "")
    content = message.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        calls = []
        for call in message.get("tool_calls") or []:
            if call.get("name"):
                calls.append({
                    "name": str(call.get("name")),
                    "args": call.get("args") if isinstance(call.get("args"), dict) else {},
                    "id": call.get("id"),
                    "type": "tool_call",
                })
        return AIMessage(content=content, tool_calls=calls)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(message.get("tool_call_id") or ""),
            name=message.get("name"),
        )
    return HumanMessage(content=str(content))


def _from_langchain_message(message) -> Dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        result = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
        if message.name:
            result["name"] = message.name
        return result
    if isinstance(message, AIMessage):
        result = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "args": call.get("args") if isinstance(call.get("args"), dict) else {},
                    "id": call.get("id"),
                    "type": "tool_call",
                }
                for call in message.tool_calls
            ]
        return result
    return {"role": "user", "content": str(message.content)}


def message_tokens(messages: List[Dict[str, Any]]) -> int:
    converted = [_to_langchain_message(message) for message in messages]
    approx = int(count_tokens_approximately(converted))
    # tokenizer 不可用时，额外计入 Unicode/CJK、工具调用结构和消息包装开销。
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(approx, (len(raw.encode("utf-8")) + 1) // 2)


def compact_messages(
    messages: List[Dict[str, Any]], max_tokens: int,
    business_state: Optional[Dict[str, Any]] = None,
    *, source_run_id: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """85% 触发、尝试压至 60%；压缩前提取显式对象和已执行工具来源。

    未压缩时原有消息字节/次序完全不变；压缩时只移除完整交互组，
    保留最新用户轮次、其 tool_calls 和所有对应 ToolMessage，不截断 JSON。
    无空间保留受保护内容/必要状态时失败关闭。
    """
    original = message_tokens(messages)
    state = state_from_messages(messages, business_state, source_run_id=source_run_id)
    if original <= int(max_tokens * .85):
        return list(messages), {
            "trimmed": False, "droppedMessages": 0, "estimatedTokens": original,
            "strategy": "stable-prefix-v2", "businessState": state,
        }

    system = [m for m in messages if m.get("role") == "system"]
    previous_snapshot = any(m.get("kind") == "context_snapshot" for m in messages)
    others = [m for m in messages if m.get("role") != "system" and m.get("kind") != "context_snapshot"]
    # Do not split a user turn from assistant tool_calls and corresponding ToolMessages.
    groups: List[List[Dict[str, Any]]] = []
    for msg in others:
        if msg.get("role") == "user" or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)
    if not groups or groups[-1][0].get("role") != "user":
        raise ApiException(429, "CONTEXT_LIMIT_EXCEEDED", "缺少当前问题，不能安全压缩")

    keep = list(groups)
    # Add a snapshot only when it carries useful state AND older messages were
    # actually discarded (or an earlier snapshot is being replaced).
    # In particular, do not spend tokens on an empty snapshot.
    while True:
        dropped_groups = len(groups) - len(keep)
        should_snapshot = has_business_context(state) and (dropped_groups > 0 or previous_snapshot)
        snapshot = [state_message(state)] if should_snapshot else []
        compacted = system + snapshot + [msg for group in keep for msg in group]
        tokens = message_tokens(compacted)
        if tokens <= int(max_tokens * .60) or len(keep) <= 1:
            break
        keep.pop(0)

    if tokens > max_tokens:
        raise ApiException(429, "CONTEXT_LIMIT_EXCEEDED", "当前问题或必要的上下文对象超出硬限制，请缩小问题")
    trimmed = compacted != messages
    return compacted, {
        "trimmed": trimmed, "droppedMessages": max(0, len(messages) - len(compacted)),
        "estimatedTokens": tokens, "beforeTokens": original,
        "businessState": state, "strategy": "complete-turns-with-state-v3",
        "reason": "HIGH_WATERMARK" if trimmed else "PROTECTED_CONTEXT",
    }


def get_run_usage(tenant_id: str, run_id: str) -> Dict[str, int]:
    """从 model_call 汇总一个 run 已经真实消耗的模型调用次数和 Token。"""
    with get_db_connection() as conn:
        row = conn.execute(
            select(
                func.count(model_call_table.c.id).label("calls"),
                func.coalesce(func.sum(model_call_table.c.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(model_call_table.c.output_tokens), 0).label("output_tokens"),
            ).where(
                and_(
                    model_call_table.c.tenant_id == tenant_id,
                    model_call_table.c.run_id == run_id,
                )
            )
        ).first()
    input_tokens = int(row.input_tokens or 0) if row else 0
    output_tokens = int(row.output_tokens or 0) if row else 0
    return {
        "modelCalls": int(row.calls or 0) if row else 0,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
    }


def assert_runtime_budget(
    tenant_id: str,
    run_id: str,
    deadline_epoch: float,
    policy: AgentRuntimePolicy,
) -> Dict[str, int]:
    # 先检查 wall-clock deadline，再检查累计调用/Token；任一命中都立即停止。
    if time.time() >= deadline_epoch:
        raise ApiException(408, "AGENT_TIMEOUT", "Agent 执行超过整体时间预算")
    usage = get_run_usage(tenant_id, run_id)
    if usage["modelCalls"] >= policy.max_model_calls:
        raise ApiException(429, "BUDGET_EXCEEDED", "Agent 模型调用次数预算已耗尽")
    if usage["totalTokens"] >= policy.max_total_tokens:
        raise ApiException(429, "BUDGET_EXCEEDED", "Agent Token 预算已耗尽")
    return usage


_RETRYABLE_CODES = {
    "MODEL_GATEWAY_UNAVAILABLE",
    "RERANK_UNAVAILABLE",
    "MILVUS_UNAVAILABLE",
    "DOCLING_UNAVAILABLE",
}


def is_retryable_exception(exc: Exception) -> bool:
    """只重试临时性网络/供应商错误；预算、取消、权限等业务错误绝不重试。"""
    if isinstance(exc, ApiException):
        if exc.code in {"BUDGET_EXCEEDED", "BATCH_BUDGET_EXCEEDED", "CANCELLED", "MODEL_PRICING_NOT_CONFIGURED"}:
            return False
        return exc.code in _RETRYABLE_CODES or exc.status_code in {429, 502, 503, 504}
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("429", "502", "503", "504", "timeout", "temporarily unavailable"))


class CircuitBreaker:
    """进程内轻量熔断器，防止依赖故障时持续重试放大流量。"""
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, float]] = {}

    def before_call(self, key: str, policy: AgentRuntimePolicy) -> None:
        """真正调用依赖前检查该 key 是否处于熔断窗口。"""
        with self._lock:
            state = self._state.get(key)
            if not state:
                return
            if state["failures"] < policy.circuit_failure_threshold:
                return
            # 熔断窗口到期后允许下一次调用试探恢复。
            if time.time() - state["opened_at"] >= policy.circuit_reset_seconds:
                self._state.pop(key, None)
                return
        raise ApiException(503, "CIRCUIT_OPEN", f"{key} 暂时熔断，请稍后重试")

    def success(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

    def failure(self, key: str, policy: AgentRuntimePolicy) -> None:
        """累计连续失败，达到阈值后打开熔断。"""
        with self._lock:
            state = self._state.setdefault(key, {"failures": 0.0, "opened_at": 0.0})
            state["failures"] += 1
            if state["failures"] >= policy.circuit_failure_threshold and not state["opened_at"]:
                state["opened_at"] = time.time()


circuit_breaker = CircuitBreaker()


def retry_call(
    key: str,
    fn: Callable[[], T],
    *,
    policy: Optional[AgentRuntimePolicy] = None,
    deadline_epoch: Optional[float] = None,
) -> T:
    policy = policy or AgentRuntimePolicy.from_env()
    circuit_breaker.before_call(key, policy)
    # call_max_attempts 包含第一次调用，不是“额外重试次数”。
    last_exc: Optional[Exception] = None

    for attempt in range(1, policy.call_max_attempts + 1):
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise ApiException(408, "AGENT_TIMEOUT", "Agent 执行超过整体时间预算")
        try:
            value = fn()
            circuit_breaker.success(key)
            return value
        except Exception as exc:
            last_exc = exc
            # 不可重试错误直接抛出；只有临时错误才进入指数退避。
            if not is_retryable_exception(exc) or attempt >= policy.call_max_attempts:
                circuit_breaker.failure(key, policy)
                raise
            # 指数退避 + 轻微 jitter，避免多个 worker 同时重试形成惊群。
            sleep_seconds = policy.retry_base_delay_seconds * (2 ** (attempt - 1))
            sleep_seconds *= 0.8 + random.random() * 0.4
            # 如果“睡完再重试”已经会超过整体 deadline，就不再浪费时间等待。
            if deadline_epoch is not None:
                remaining = deadline_epoch - time.time()
                if remaining <= sleep_seconds:
                    raise ApiException(408, "AGENT_TIMEOUT", "Agent 执行超过整体时间预算") from exc
            time.sleep(sleep_seconds)

    assert last_exc is not None
    raise last_exc


_NON_RETRYABLE_JOB_CODES = {
    "BATCH_BUDGET_EXCEEDED",
    "MODEL_PRICING_NOT_CONFIGURED",
    "CANCELLED",
    "USER_CANCELLED",
    "AGENT_TIMEOUT",
    "BUDGET_EXCEEDED",
    "REPEATED_CALLS",
    "INVALID_INPUT",
    "INVALID_TOOL_ARGUMENTS",
    "UNKNOWN_TOOL",
    "OPERATION_FORBIDDEN",
    "SOURCE_CHANGED",
    "VERIFICATION_FAILED",
    "INVALID_MODEL_OUTPUT",
    "APPROVAL_REQUIRED",
    "REQUESTER_NOT_ACTIVE",
}


def is_retryable_job_code(code: Optional[str]) -> bool:
    """Worker 失败后是否允许重新入队；业务确定性失败全部列入禁止重试集合。"""
    if not code:
        return True
    return str(code) not in _NON_RETRYABLE_JOB_CODES
