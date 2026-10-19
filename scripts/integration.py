#!/usr/bin/env python3
"""Opt-in, model-free HTTP integration acceptance across authorization boundaries."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

from quickstart import ROOT
from tooling import (ApiResponse, CheckError, HttpClient, assert_http, assert_that, encode_multipart,
                     local_quote, log_error, report_json, seed_password)


def self_test() -> None:
    assert_http(ApiResponse(409, {"code": "EXPECTED"}, ""), 409, "EXPECTED")
    for bad in (ApiResponse(200, {"code": "EXPECTED"}, ""),
                ApiResponse(500, {"code": "EXPECTED"}, ""),
                ApiResponse(409, {"code": "WRONG"}, "")):
        try:
            assert_http(bad, 409, "EXPECTED")
        except CheckError:
            continue
        raise CheckError("assert_http accepted invalid response")
    try:
        assert_that(False, "missing marker")
    except CheckError:
        print("Integration assertion self-test passed")
        return
    raise CheckError("False negative was accepted")


def wait_for_job(client: HttpClient, job_id: str) -> dict:
    for _ in range(30):
        time.sleep(2)
        job = client.ok("GET", f"/api/v1/admin/jobs/{job_id}")
        if job["status"] == "COMPLETED":
            return job
        assert_that(job["status"] not in ("FAILED", "CANCELLED", "NEEDS_REVIEW"),
                    f"Ingestion stopped: {job['status']}, code {job.get('errorCode')}")
    raise CheckError("Ingestion did not complete in 60 seconds")


def integration(base_url: str, environment: str) -> None:
    password = seed_password(environment)
    admin, ops = HttpClient(base_url), HttpClient(base_url)
    assert_http(admin.request("GET", "/api/v1/auth/csrf"), 200, label="backend readiness")
    admin.login(environment, "admin", password)
    ops.login(environment, "ops", password)
    admin_me = admin.ok("GET", "/api/v1/me")
    ops_me = ops.ok("GET", "/api/v1/me")
    assert_that(admin_me.get("admin") and not ops_me.get("admin"), "Admin/ops roles incorrect")
    suffix = uuid.uuid4().hex[:10]
    kb = admin.ok("POST", "/api/v1/admin/knowledge-bases",
                  {"name": f"verification-{suffix}",
                   "readers": [{"type": "USER", "userId": admin_me["userId"]}]},
                  admin.headers())
    kid = kb["id"]
    acl = admin.ok("GET", f"/api/v1/admin/knowledge-bases/{kid}/acl")
    admin.ok("PUT", f"/api/v1/admin/knowledge-bases/{kid}/acl",
             {"expectedVersion": acl["rowVersion"],
              "readers": [{"type": "USER", "userId": admin_me["userId"]},
                          {"type": "USER", "userId": ops_me["userId"]}]},
             admin.headers())

    marker = f"KBVERIFY{suffix}"
    doc_path = ROOT / ".runtime" / f"verification-{suffix}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(f"# 验收资料\n\n系统核验标记为 {marker}，端口为 3307。\n", encoding="utf-8")
    build_config = {"embedding": False, "summary": False, "questions": False, "vision": False}
    body = encode_multipart(
        {"kbId": kid, "buildConfig": json.dumps(build_config)},
        doc_path.name, doc_path.read_bytes())
    uploaded = admin.request("POST", "/api/v1/admin/documents", headers=admin.headers(), multipart=body)
    assert_http(uploaded, 202, label="document ingestion")
    document_id, version_id, job_id = (uploaded.body[k] for k in ("documentId", "versionId", "jobId"))
    wait_for_job(admin, job_id)
    query_path = f"/api/v1/search?query={local_quote(marker)}&kbIds={local_quote(kid)}"
    admin_search = admin.ok("GET", query_path)
    assert_that(any(r.get("documentId") == document_id and marker in (r.get("text") or "")
                    for r in admin_search["evidence"]), "Admin BM25 did not retrieve marker")
    assert_that(not ops.ok("GET", query_path)["evidence"], "Ops retrieved ungranted document")
    assert_http(ops.request("GET", f"/api/v1/documents/{document_id}"), 404,
                "RESOURCE_NOT_FOUND", "ungranted document")
    own_doc = admin.ok("GET", f"/api/v1/documents/{document_id}")
    grant = admin.ok("PUT", f"/api/v1/admin/documents/{document_id}/acl",
                     {"expectedVersion": own_doc["rowVersion"], "mode": "RESTRICT",
                      "readers": [{"type": "USER", "userId": ops_me["userId"]}]}, admin.headers())
    assert_http(ops.request("GET", f"/api/v1/documents/{document_id}"), 200,
                label="document after grant")
    granted_hits = ops.ok("GET", query_path)["evidence"]
    assert_that(any(r.get("documentId") == document_id for r in granted_hits),
                "Granted document not retrieved")
    admin.ok("PUT", f"/api/v1/admin/documents/{document_id}/acl",
             {"expectedVersion": grant["rowVersion"], "mode": "RESTRICT",
              "readers": [{"type": "USER", "userId": admin_me["userId"]}]}, admin.headers())
    assert_that(not ops.ok("GET", query_path)["evidence"], "Document retrieved after revocation")
    assert_http(ops.request("GET", f"/api/v1/documents/{document_id}"), 404,
                "RESOURCE_NOT_FOUND", "document after revocation")
    if environment == "demo":
        isolation = admin.request("POST", "/api/v1/admin/eval/datasets",
                                  {"name": f"must-be-rejected-{suffix}"}, admin.headers())
        assert_http(isolation, 409, "EVALUATION_ENVIRONMENT_REQUIRED", "evaluation isolation")

    evaluation_run_id, evaluation_metrics = None, None
    if environment == "evaluation":
        ds = admin.ok("POST", "/api/v1/admin/eval/datasets",
                      {"name": f"http-verification-{suffix}"}, admin.headers())
        did = ds["id"]
        case = {"caseKey": f"HTTP-{suffix}", "groupKey": f"HTTP-{suffix}", "split": "TEST",
                "category": "FACT", "question": {"text": f"系统核验标记 {marker} 是什么？"},
                "persona": {"loginName": "admin"}, "expected": {
                    "status": "ANSWER", "evidenceGroups": [{
                        "fact": "核验标记",
                        "alternatives": [{"documentId": document_id, "versionId": version_id,
                                          "quote": f"系统核验标记为 {marker}", "relevanceGrade": 3}]
                    }]}, "annotationStatus": "AUTO_REVIEWED"}
        admin.ok("POST", f"/api/v1/admin/eval/dataset-versions/{did}/cases",
                 {"cases": [case]}, admin.headers())
        admin.ok("POST", f"/api/v1/admin/eval/dataset-versions/{did}/freeze", {}, admin.headers())
        snapshot = admin.ok("POST", "/api/v1/admin/eval/snapshots",
                            {"name": f"http-verification-{suffix}", "documentIds": [document_id],
                             "approvedSyntheticOrRedacted": True}, admin.headers())
        configs = admin.ok("GET", "/api/v1/admin/configs")
        baseline = next((c for c in configs if c.get("kind") == "RETRIEVAL" and
                         c.get("name") == "baseline" and c.get("status") == "PUBLISHED"), None)
        assert_that(baseline, "Published baseline config missing")
        request = {"datasetVersionId": did, "snapshotId": snapshot["snapshotId"],
                   "configId": baseline["id"], "split": "TEST", "repeats": 1,
                   "mode": "RETRIEVAL_ONLY", "strict": False}
        estimate = admin.ok("POST", "/api/v1/admin/eval/estimates", request, admin.headers())
        assert_that(estimate.get("estimatedCostCny") == 0, "Retrieval-only estimate not zero")
        request.update({"estimateId": estimate["estimateId"], "maxCostCny": 1})
        started = admin.ok("POST", "/api/v1/admin/eval/runs", request, admin.headers())
        evaluation_run_id = started["runId"]
        for _ in range(30):
            time.sleep(1)
            evaluation = admin.ok("GET", f"/api/v1/admin/eval/runs/{evaluation_run_id}")
            if evaluation["status"] not in ("QUEUED", "RUNNING"):
                break
        assert_that(evaluation["status"] == "COMPLETED", "Retrieval evaluation did not finish")
        evaluation_metrics = evaluation["metrics"]
        for metric in ("hitAt5", "recallAt5", "mrr", "ndcgAt5"):
            assert_that(evaluation_metrics.get(metric) == 1, f"Expected {metric}=1")
        for metric in ("retrievalDenominator", "ndcgDenominator"):
            assert_that(evaluation_metrics.get(metric) == 1, f"Expected {metric}=1")

    checks = ["admin and ops authentication", "model-free markdown ingestion",
              "BM25 positive retrieval", "document read denial before grant",
              "document ACL grant", "document ACL revocation"]
    if environment == "demo":
        checks.append("evaluation isolation in demo")
    if environment == "evaluation":
        checks.append("frozen one-case retrieval evaluation with Hit, Recall, MRR and nDCG")
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(), "environment": environment,
        "scope": "No-model HTTP integration", "checks": checks, "modelCalls": 0,
        "notTested": ["generated answers", "real-model quality", "Docling PDF or DOCX parsing",
                      "complete acceptance-criteria matrix"],
        "documentId": document_id, "jobId": job_id, "evaluationRunId": evaluation_run_id,
        "evaluationMetrics": evaluation_metrics,
    }
    path = ROOT / ".runtime" / "reports" / "http-integration.json"
    report_json(path, report)
    print(f"No-model HTTP integration passed. Report: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--environment", choices=("demo", "test", "evaluation"), default="demo")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            integration(args.base_url, args.environment)
    except (CheckError, ValueError, KeyError, OSError) as exc:
        return log_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
