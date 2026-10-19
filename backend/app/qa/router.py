import json
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Response, status
from fastapi.responses import StreamingResponse
from app.identity.deps import Actor, get_current_actor
from app.qa.service import QaService

router = APIRouter(tags=["qa"])


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    return QaService.create_conversation(actor, body)


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session_alias(
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    """Compatibility alias for contract tests calling /api/v1/sessions."""
    return QaService.create_conversation(actor, body)


@router.get("/conversations")
def list_conversations(
    actor: Actor = Depends(get_current_actor),
):
    return QaService.list_conversations(actor)


@router.get("/conversations/{conversation_id}")
def get_conversation_detail(
    conversation_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.get_conversation_detail(actor, conversation_id)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    actor: Actor = Depends(get_current_actor),
):
    QaService.delete_conversation(actor, conversation_id)
    return {"status": "DELETED"}


@router.post("/conversations/{conversation_id}/segments")
def create_segment(
    conversation_id: str,
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    return QaService.create_segment(actor, conversation_id, body)


@router.post("/conversations/{conversation_id}/runs", status_code=status.HTTP_202_ACCEPTED)
def submit_run(
    conversation_id: str,
    body: Dict[str, Any],
    response: Response,
    background_tasks: BackgroundTasks,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    actor: Actor = Depends(get_current_actor),
):
    response.status_code = status.HTTP_202_ACCEPTED
    result = QaService.submit_run(actor, conversation_id, body, idempotency_key)
    # 新请求统一 Durable Agent，旧 REALTIME 代码仅供旧任务兼容。
    # 已有幂等结果不得再次调度，以免两个请求同时重复生成。
    from app.config import is_test_env
    if result.get("executionMode") == "REALTIME" and not result.get("reused") and not is_test_env():
        background_tasks.add_task(QaService.execute_queued_run, actor, result["runId"])
    return result


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.get_run(actor, run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.cancel_run(actor, run_id)

@router.post("/runs/{run_id}/approvals")
def resolve_tool_approval(
    run_id: str,
    body: Dict[str, Any],
    actor: Actor = Depends(get_current_actor),
):
    return QaService.resolve_tool_approval(actor, run_id, body)


@router.get("/runs/{run_id}/events")
def stream_run_events(
    run_id: str,
    actor: Actor = Depends(get_current_actor),
):
    # 流式响应开始前先完成授权检查；一旦开始输出 SSE/stream，就不能再依赖后续 HTTP 状态码阻止越权内容。
    QaService.get_run(actor, run_id)

    def event_stream():
        last_sequence = 0
        terminal = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED"}
        idle_after_terminal = 0
        while True:
            snapshot = QaService.get_event_snapshot(actor, run_id, last_sequence)
            for event in snapshot["events"]:
                last_sequence = max(last_sequence, int(event["sequenceNo"]))
                yield "event: run_event\n"
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            if snapshot["status"] in terminal:
                if not snapshot["events"]:
                    idle_after_terminal += 1
                if idle_after_terminal >= 2:
                    yield "event: done\n"
                    yield "data: " + json.dumps({"status": snapshot["status"]}) + "\n\n"
                    return
            time.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.get("/runs/{run_id}/evidence/{evidence_id}")
def get_evidence(
    run_id: str,
    evidence_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.get_evidence(actor, run_id, evidence_id)


@router.get("/admin/runs/{run_id}/trace")
def get_trace(
    run_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.get_trace(actor, run_id)


@router.put("/runs/{run_id}/feedback")
def submit_feedback(
    run_id: str,
    body: Dict[str, Any],
    actor: Actor = Depends(get_current_actor),
):
    return QaService.submit_feedback(actor, run_id, body)


@router.get("/admin/feedback")
def admin_feedback_list(
    actor: Actor = Depends(get_current_actor),
):
    return QaService.admin_feedback_list(actor)


@router.get("/admin/feedback/{feedback_id}")
def admin_feedback_detail(
    feedback_id: str,
    actor: Actor = Depends(get_current_actor),
):
    return QaService.admin_feedback_detail(actor, feedback_id)


@router.patch("/admin/feedback/{feedback_id}")
def admin_feedback_update(
    feedback_id: str,
    body: Dict[str, Any],
    actor: Actor = Depends(get_current_actor),
):
    return QaService.admin_feedback_update(actor, feedback_id, body)

