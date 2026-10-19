import base64
import hashlib
import hmac
import json
import os
import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, Request

from app.errors import ApiException
from app.identity.deps import Actor, require_admin
from app.model_budget import BudgetService


router = APIRouter(tags=["budget"])


def _internal_actor(request: Request) -> Actor:
    token = os.getenv("RAG_INTERNAL_BUDGET_TOKEN", "")
    tenant_id = os.getenv("RAG_INTERNAL_BUDGET_TENANT", "")
    supplied = request.headers.get("authorization", "")
    if (
        len(token) < 16
        or not tenant_id
        or request.headers.get("origin") is not None
        or not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8"))
    ):
        raise ApiException(403, "OPERATION_FORBIDDEN", "内部服务认证失败")
    return Actor(tenant_id=tenant_id, user_id="", admin=False, security_epoch=0)


def _sign(reservation_id: str, pricing: dict) -> str:
    token = os.getenv("RAG_INTERNAL_BUDGET_TOKEN", "")
    raw = reservation_id + json.dumps(pricing, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hmac.new(token.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@router.get("/admin/budget")
def budget_summary(actor: Actor = Depends(require_admin)):
    return BudgetService.summary(actor.tenant_id)


@router.post("/internal/budget/reservations")
def reserve_budget(request: Request, body: dict):
    actor = _internal_actor(request)
    expected_caller = os.getenv("RAG_INTERNAL_BUDGET_CALLER", "evaluation")
    caller = str(body.get("callerEnvironment", ""))
    if caller != expected_caller:
        raise ApiException(403, "OPERATION_FORBIDDEN", "未知预算调用方")

    call_id = str(body.get("callId", ""))
    try:
        uuid.UUID(call_id)
    except Exception as exc:
        raise ApiException(400, "INVALID_INPUT", "调用 ID 无效") from exc

    max_input = int(body.get("maxInputTokens", -1))
    max_output = int(body.get("maxOutputTokens", -1))
    if max_input < 0 or max_input > 120000 or max_output < 0 or max_output > 6000:
        raise ApiException(400, "INVALID_INPUT", "预留 token 无效")

    model_alias = str(body.get("modelAlias") or "chat-main")
    row = BudgetService.reserve(
        tenant_id=actor.tenant_id,
        call_id=call_id,
        model_alias=model_alias,
        input_upper_bound=max_input,
        output_upper_bound=max_output,
        caller_environment=caller,
    )
    pricing = {
        "currency": "CNY",
        "modelAlias": model_alias,
        "inputPricePerMillion": str(row["pricing"]["inputPricePerMillion"]),
        "outputPricePerMillion": str(row["pricing"]["outputPricePerMillion"]),
    }
    return {
        "reservationId": row["id"],
        "status": row["status"],
        "pricingSnapshot": pricing,
        "pricingSignature": _sign(row["id"], pricing),
        "reservedAmount": str(row["amount"]),
    }


@router.post("/internal/budget/reservations/{reservation_id}/settle")
def settle_budget(request: Request, reservation_id: str, body: dict):
    actor = _internal_actor(request)
    status = str(body.get("status", ""))
    actual = None

    if status == "SETTLED":
        input_tokens = int(body.get("inputTokens", -1))
        output_tokens = int(body.get("outputTokens", -1))
        if input_tokens < 0 or output_tokens < 0:
            raise ApiException(400, "INVALID_INPUT", "结算需要实际用量")
        pricing = body.get("pricingSnapshot")
        if not isinstance(pricing, dict):
            raise ApiException(400, "INVALID_INPUT", "缺少 pricingSnapshot")
        supplied_signature = str(body.get("pricingSignature", ""))
        if not hmac.compare_digest(
            supplied_signature.encode("utf-8"),
            _sign(reservation_id, pricing).encode("utf-8"),
        ):
            raise ApiException(403, "OPERATION_FORBIDDEN", "价格快照认证失败")
        actual = BudgetService.estimate(pricing, input_tokens, output_tokens)

    BudgetService.settle(actor.tenant_id, reservation_id, status, actual)
    return {"reservationId": reservation_id, "status": status}
