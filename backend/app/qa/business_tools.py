"""只用于演示的业务诊断工具。返回内容不代表真实 Java 业务系统数据。

fixture 均为固定合成数据，输入对象只原样标识本次模拟问题，不进行资格计算或接口调用。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.errors import ApiException

FIXTURE_VERSION = "business-demo-v1"
MOCK_NOTICE = "模拟结果，未查询真实业务系统"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_TOOL_SPECS = [
    ("diagnose_coupon", "模拟诊断用户为什么不能使用指定优惠券；必须传明确 user_id 和 coupon_id。", ["user_id", "coupon_id"]),
    ("get_final_price_rules", "读取固定演示到手价计算规则，不需要用户 ID、购物车或商品参数。", []),
    ("diagnose_product_group", "模拟诊断商品组为什么没有向指定用户出品；必须传明确 user_id 和 product_group_id。", ["user_id", "product_group_id"]),
    ("diagnose_subsidy", "模拟诊断用户为什么未命中补贴活动；必须传明确 user_id 和 activity_id。", ["user_id", "activity_id"]),
]
BUSINESS_TOOL_NAMES = frozenset(spec[0] for spec in _TOOL_SPECS)

BUSINESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description + " 输出仅为模拟结果，未查询真实业务系统。",
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string", "minLength": 1, "maxLength": 128,
                                     "description": "必须是用户明确提供的对象 ID，不得编造"} for key in required},
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    for name, description, required in _TOOL_SPECS
]


def _fen(value: int) -> str:
    return f"{value // 100}.{value % 100:02d}元"


def execute_business_tool(
    name: str,
    args: Dict[str, Any],
    *,
    queried_at: Optional[datetime] = None,
    fixture_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """验证参数后执行固定 fixture；测试可显式注入不同结果，绝不从自然语言触发分支。"""
    if name not in BUSINESS_TOOL_NAMES:
        raise ApiException(400, "UNKNOWN_TOOL", "不支持此工具")
    required = next(req for tool_name, _, req in _TOOL_SPECS if tool_name == name)
    if not isinstance(args, dict) or set(args) != set(required) or any(not isinstance(args.get(key), str) or not args[key].strip() for key in required):
        missing = [key for key in required if not isinstance((args or {}).get(key), str)
                   or not str(args[key]).strip()] if isinstance(args, dict) else required
        message = "请补充：" + "、".join(missing) if missing else "业务工具参数不匹配"
        raise ApiException(400, "INVALID_TOOL_ARGUMENTS", message)
    for key in required:
        if not _ID_PATTERN.fullmatch(args[key]):
            raise ApiException(400, "INVALID_TOOL_ARGUMENTS", f"请提供有效的 {key}")
    query_objects = {key: args[key] for key in required}
    if name == "diagnose_coupon":
        data = {
            "decision": "UNAVAILABLE", "failedCondition": "ELIGIBLE_AMOUNT_BELOW_THRESHOLD",
            "eligibleAmountFen": 16800, "thresholdFen": 20000, "excludedAmountFen": 7000,
            "explanation": f"参与金额{_fen(16800)}未达到门槛{_fen(20000)}；排除商品金额{_fen(7000)}。",
        }
    elif name == "get_final_price_rules":
        data = {
            "rule": "先减活动优惠，再检查并扣券，再扣补贴，最低为零；仅为演示规则。",
            "order": ["商品金额", "活动优惠", "优惠券", "补贴"],
            "exampleFen": {"amount": 20000, "promotion": 2000, "coupon": 3000,
                           "subsidy": 1000, "final": 14000},
            "example": "200-20-30-10=140元",
        }
    elif name == "diagnose_product_group":
        data = {
            "decision": "NOT_EXPOSED", "groupEnabled": True, "hasCandidates": True,
            "failedCondition": "USER_REGION_NOT_TARGETED",
            "explanation": "商品组已启用且存在候选商品，但用户地区不在投放范围。",
        }
    else:
        data = {
            "decision": "NOT_ELIGIBLE", "activityValid": True,
            "failedCondition": "NOT_NEW_CUSTOMER",
            "explanation": "补贴活动有效，但用户不满足新客资格。",
        }
    if fixture_override is not None:
        # 只供确定性测试使用的注入点；不能由用户请求影响 fixture。
        data = dict(fixture_override)
    queried = queried_at or datetime.now(timezone.utc)
    if queried.tzinfo is None:
        queried = queried.replace(tzinfo=timezone.utc)
    return {
        "status": "OK",
        "data": data,
        "mock": True,
        "fixtureVersion": FIXTURE_VERSION,
        "queryObjects": query_objects,
        "queriedAt": queried.isoformat(),
        "notice": MOCK_NOTICE,
    }


def business_evidence_text(name: str, result: Dict[str, Any]) -> str:
    """只从确定性数据生成模型可引用的原文，金额不交给模型二次计算。"""
    data = result["data"]
    objects = result["queryObjects"]
    scope = "；".join(f"{key}={value}" for key, value in objects.items()) or "无查询对象"
    return (
        f"{MOCK_NOTICE}。\n"
        f"工具：{name}；查询对象：{scope}；fixtureVersion：{result['fixtureVersion']}；"
        f"查询时间：{result['queriedAt']}。\n"
        f"演示结果：{data['explanation'] if 'explanation' in data else data['rule'] + ' 示例：' + data['example']}\n"
        f"结论或失败条件：{data.get('failedCondition', data.get('decision', 'RULE_ONLY'))}。"
    )
