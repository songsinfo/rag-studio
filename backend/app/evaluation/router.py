import uuid
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, Header, Path, Query, status

from app.identity.deps import Actor, get_current_actor
from app.evaluation.catalog import EvaluationCatalog
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/admin/eval", tags=["evaluation"])


@router.get("/datasets")
def datasets(actor: Actor = Depends(get_current_actor)):
    return EvaluationCatalog.datasets(actor)


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(body: Dict[str, Any], actor: Actor = Depends(get_current_actor)):
    return EvaluationCatalog.create_dataset(actor, body)


@router.post("/datasets/{id}/versions", status_code=status.HTTP_201_CREATED)
def create_version(
    id: str = Path(...),
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    source_id = str(body.get("sourceVersionId") or id).strip()
    return EvaluationCatalog.copy_version(actor, source_id)


@router.get("/dataset-versions/{versionId}/cases")
def get_cases(
    versionId: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationCatalog.cases(actor, versionId)


@router.post("/dataset-versions/{versionId}/cases")
def save_cases(
    versionId: str = Path(...),
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    if "cases" in body and isinstance(body["cases"], list):
        case_list = body["cases"]
    else:
        case_list = [body]
    return EvaluationCatalog.save_cases(actor, versionId, case_list)


@router.post("/dataset-versions/{versionId}/freeze")
def freeze_dataset(
    versionId: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationCatalog.freeze(actor, versionId)


@router.patch("/cases/{id}")
def update_case(
    id: str = Path(...),
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationCatalog.update_case(actor, id, body)


@router.delete("/cases/{id}")
def delete_case(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    EvaluationCatalog.delete_case(actor, id)
    return {"status": "DELETED"}


@router.get("/snapshots")
def snapshots(actor: Actor = Depends(get_current_actor)):
    return EvaluationCatalog.snapshots(actor)


@router.post("/snapshots", status_code=status.HTTP_201_CREATED)
def create_snapshot(
    body: Dict[str, Any],
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationCatalog.create_snapshot(actor, body)


@router.post("/estimates")
def estimate(
    body: Dict[str, Any],
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.estimate(actor, body)


@router.post("/runs", status_code=status.HTTP_201_CREATED)
def start_run(
    body: Dict[str, Any],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    actor: Actor = Depends(get_current_actor),
):
    eff_key = idempotency_key if idempotency_key and idempotency_key.strip() else str(uuid.uuid4())
    return EvaluationRunner.start(actor, body, eff_key)


@router.get("/runs/{id}")
def get_run(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.get_run(actor, id)


@router.get("/runs/{id}/results")
def run_results(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.results(actor, id)


@router.post("/runs/{id}/cancel")
def cancel_run(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.cancel(actor, id)


@router.post("/runs/{id}/exports")
def export_run(
    id: str = Path(...),
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.export_run(actor, id, body)


@router.post("/results/{id}/reviews", status_code=status.HTTP_201_CREATED)
def review_result(
    id: str = Path(...),
    body: Dict[str, Any] = {},
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.review(actor, id, body)


@router.get("/comparisons")
def compare_runs(
    runIds: str = Query(..., description="Comma-separated run IDs"),
    actor: Actor = Depends(get_current_actor),
):
    ids = [s.strip() for s in runIds.split(",") if s.strip()]
    return EvaluationRunner.compare(actor, ids)


@router.get("/runs/{id}/baseline-diff")
def baseline_diff(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    return EvaluationRunner.compare_with_baseline(actor, id)

@router.post("/runs/{id}/langfuse-scores")
def sync_langfuse_scores(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    """Opt-in administrator action; does not block normal Judge/review paths."""
    from app.evaluation.langfuse_sync import sync_evaluation_scores
    return sync_evaluation_scores(actor, id)


@router.post("/runs/{id}/assertion-diagnostics", status_code=status.HTTP_202_ACCEPTED)
def start_assertion_diagnostic(body: Dict[str, Any], id: str = Path(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    actor: Actor = Depends(get_current_actor)):
    from app.evaluation.assertion_diagnostic import start_diagnostic
    return start_diagnostic(actor, id, body, idempotency_key)


@router.get("/assertion-diagnostics/{jobId}")
def assertion_diagnostic(jobId: str = Path(...), actor: Actor = Depends(get_current_actor)):
    from app.evaluation.assertion_diagnostic import get_diagnostic
    return get_diagnostic(actor, jobId)


@router.post("/assertion-diagnostics/{jobId}/calibration")
def assertion_calibration(body: Dict[str, Any], jobId: str = Path(...), actor: Actor = Depends(get_current_actor)):
    from app.evaluation.assertion_diagnostic import calibrate_diagnostic
    return calibrate_diagnostic(actor, jobId, body)

@router.get("/qa-runs/{id}/retrieval-diagnostics")
def get_private_retrieval_diagnostics(
    id: str = Path(...),
    actor: Actor = Depends(get_current_actor),
):
    """Read protected retrieval artifact after persona and snapshot ACL checks."""
    from app.evaluation.diagnostics import read_eval_diagnostics
    return read_eval_diagnostics(actor, id)
