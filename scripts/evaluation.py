#!/usr/bin/env python3
"""Validate/bind synthetic evaluation datasets and run opt-in B0-B3 ablations."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from quickstart import ROOT
from tooling import CheckError, HttpClient, assert_that, local_quote, report_json, seed_password

SEEDED = {"admin", "ops", "product", "engineer"}
SPLITS = {"DEV", "TEST"}
STATUSES = {"ANSWER", "PARTIAL", "ABSTAIN", "CONFLICT"}
ANNOTATIONS = {"DRAFT", "AUTO_REVIEWED", "HUMAN_REVIEWED"}


def dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert_that(bool(rows), "Evaluation dataset is empty")
    return rows


def validate(path: Path, expected_count: int = 0, source_root: Path | None = None,
             require_runtime_bindings: bool = False) -> dict:
    cases = dataset(path)
    if expected_count:
        assert_that(len(cases) == expected_count, f"Expected {expected_count} cases, got {len(cases)}")
    keys: set[str] = set()
    groups: dict[str, str] = {}
    splits: dict[str, int] = {}
    categories: dict[str, int] = {}
    personas: dict[str, int] = {}
    sources: set[str] = set()
    runtime_ready = True
    for row in cases:
        for key in ("caseKey", "groupKey", "split", "category", "question", "persona", "expected", "annotationStatus"):
            assert_that(key in row and row[key] is not None, f"Case missing {key}")
        key = str(row["caseKey"])
        assert_that(bool(key.strip()) and key not in keys, f"Empty/duplicate caseKey: {key}")
        keys.add(key)
        split = str(row["split"]).upper()
        assert_that(split in SPLITS, f"{key}: invalid split")
        group = str(row["groupKey"])
        assert_that(group not in groups or groups[group] == split, f"{group}: DEV/TEST leakage")
        groups[group] = split
        splits[split] = splits.get(split, 0) + 1
        category = str(row["category"])
        assert_that(bool(category.strip()), f"{key}: empty category")
        categories[category] = categories.get(category, 0) + 1
        question = row["question"]
        assert_that(bool(str(question.get("text") or "").strip()) or bool(question.get("turns")),
                    f"{key}: empty question")
        persona = str(row["persona"]["loginName"])
        assert_that(bool(persona.strip()), f"{key}: empty persona")
        personas[persona] = personas.get(persona, 0) + 1
        runtime_ready &= persona in SEEDED
        status = str(row["expected"]["status"]).upper()
        assert_that(status in STATUSES, f"{key}: invalid status")
        assert_that(str(row["annotationStatus"]).upper() in ANNOTATIONS, f"{key}: invalid annotation")
        evidence = row["expected"].get("evidenceGroups") or []
        assert_that(status == "ABSTAIN" or bool(evidence), f"{key}: no evidence groups")
        for evidence_group in evidence:
            assert_that(bool(str(evidence_group.get("fact") or "").strip()), f"{key}: empty fact")
            alternatives = evidence_group.get("alternatives") or []
            assert_that(bool(alternatives), f"{key}: empty alternatives")
            for anchor in alternatives:
                docid, version, quote = (str(anchor.get(k) or "") for k in ("documentId", "versionId", "quote"))
                assert_that(bool(docid and version and quote), f"{key}: incomplete anchor")
                grade = anchor.get("relevanceGrade")
                assert_that(type(grade) is int and 0 <= grade <= 3, f"{key}: invalid relevanceGrade")
                sources.add(docid)
                try:
                    uuid.UUID(docid)
                    uuid.UUID(version)
                except ValueError:
                    runtime_ready = False
                if source_root is not None:
                    root = source_root.resolve(strict=True)
                    target = (root / docid).resolve(strict=True)
                    assert_that(target.is_relative_to(root) and target.is_file(), f"{key}: invalid source path")
                    assert_that(quote in target.read_text(encoding="utf-8"), f"{key}: quote not found in source")
    report = {
        "path": str(path.resolve()), "cases": len(cases), "splits": splits,
        "categories": categories, "personas": personas,
        "uniqueSourceRefs": len(sources), "sourceQuotesChecked": source_root is not None,
        "runtimeBindingsReady": runtime_ready,
        "runtimeBindingNote": (
            "All personas and evidence IDs use seeded accounts and UUID-shaped runtime IDs."
            if runtime_ready else
            "Template is structurally valid; map source and persona aliases to runtime records."
        ),
    }
    assert_that(not require_runtime_bindings or runtime_ready, "Unresolved runtime bindings")
    return report


def bind(input_path: Path, binding_path: Path, output_path: Path) -> None:
    rows = dataset(input_path)
    if not binding_path.exists():
        personas: dict[str, str] = {}
        sources: dict[str, dict[str, str]] = {}
        for row in rows:
            login = str(row["persona"]["loginName"])
            if login not in SEEDED:
                personas[login] = "ops" if login == "operator" else ""
            for group in row["expected"].get("evidenceGroups") or []:
                for anchor in group.get("alternatives") or []:
                    sources.setdefault(str(anchor["documentId"]), {"documentId": "", "versionId": ""})
        report_json(binding_path, {"personas": personas, "sources": sources})
        raise CheckError(f"Binding template created: {binding_path}. Fill values and rerun")
    bindings = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    assert_that(isinstance(bindings.get("personas"), dict)
                and isinstance(bindings.get("sources"), dict), "Binding file needs personas/sources maps")
    result = copy.deepcopy(rows)
    for row in result:
        login = row["persona"]["loginName"]
        if login not in SEEDED:
            login = bindings["personas"].get(login)
            assert_that(login in SEEDED, f"Missing seeded persona binding for {login}")
            row["persona"]["loginName"] = login
        for group in row["expected"].get("evidenceGroups") or []:
            for anchor in group.get("alternatives") or []:
                ref = anchor["documentId"]
                mapped = bindings["sources"].get(ref)
                assert_that(isinstance(mapped, dict), f"Missing source binding for {ref}")
                for name in ("documentId", "versionId"):
                    value = str(mapped.get(name) or "")
                    try:
                        uuid.UUID(value)
                    except ValueError as exc:
                        raise CheckError(f"{name} for {ref} is not a UUID") from exc
                    anchor[name] = value
    # Validate content before writing anything.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pending.jsonl")
    try:
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf-8")
        validate(temporary, expected_count=len(rows), require_runtime_bindings=True)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Bound {len(rows)} cases: {output_path}")


def configurations(path: Path) -> list[dict]:
    configs = json.loads(path.read_text(encoding="utf-8"))
    assert_that([c["id"] for c in configs] == ["B0", "B1", "B2", "B3"], "Expected ordered B0-B3")
    for index, c in enumerate(configs):
        payload = c["payload"]
        expected_mode = ("BM25", "DENSE", "HYBRID", "HYBRID")[index]
        assert_that(payload["retrieval"]["mode"] == expected_mode, f"{c['id']}: wrong mode")
        assert_that(bool(payload["post"]["rerank"]) == (index == 3), f"{c['id']}: rerank mismatch")
        assert_that(all(not payload["query"].get(k) for k in ("rewrite", "hyde", "multiQuery")),
                    f"{c['id']}: query expansions enabled")
        assert_that(payload["retrieval"]["representations"] == ["ORIGINAL"], "Only ORIGINAL vectors allowed")
        for section, name, value in (("retrieval", "perRouteK", 50), ("retrieval", "numCandidates", 200),
                                     ("fusion", "rrfK", 60), ("post", "rerankTopN", 30)):
            assert_that(payload[section][name] == value, f"{c['id']}: {name} drift")
        assert_that(not payload["post"]["mmr"] and not payload["post"]["parent"], "MMR/parent must be off")
    return configs


def ablation(args: argparse.Namespace) -> None:
    configs = configurations(args.config_path)
    if args.self_test:
        print("Self-test passed: B0-B3 config invariants")
        return
    assert_that(0 < args.max_cost_cny <= 10 and math.isfinite(args.max_cost_cny), "Invalid cost cap")
    assert_that(args.timeout_seconds >= 60, "Timeout must be at least 60 seconds")
    provision = args.dataset_path is not None
    assert_that((provision and not args.dataset_version_id and not args.snapshot_id
                 and args.approved_synthetic_or_redacted) or
                (not provision and args.dataset_version_id and args.snapshot_id),
                "Provide approved --dataset-path OR both existing --dataset-version-id/--snapshot-id")
    client = HttpClient(args.base_url, timeout=60)
    client.login("evaluation", "admin", seed_password("evaluation"))
    started_at = datetime.now(timezone.utc).isoformat()
    execution_id = uuid.uuid4().hex
    dataset_id, snapshot_id = args.dataset_version_id, args.snapshot_id
    if provision:
        validate(args.dataset_path, require_runtime_bindings=True)
        cases = dataset(args.dataset_path)
        created = client.ok("POST", "/api/v1/admin/eval/datasets",
                            {"name": args.dataset_name}, client.headers())
        dataset_id = created["id"]
        client.ok("POST", f"/api/v1/admin/eval/dataset-versions/{dataset_id}/cases",
                  {"cases": cases}, client.headers())
        client.ok("POST", f"/api/v1/admin/eval/dataset-versions/{dataset_id}/freeze", {}, client.headers())
        ids = sorted({a["documentId"] for r in cases for g in r["expected"].get("evidenceGroups", [])
                      for a in g["alternatives"]})
        assert_that(bool(ids), "No evidence documents for snapshot")
        snapshot = client.ok("POST", "/api/v1/admin/eval/snapshots",
                             {"name": f"{args.dataset_name}-{execution_id}", "documentIds": ids,
                              "approvedSyntheticOrRedacted": True}, client.headers())
        snapshot_id = snapshot["snapshotId"]
    runs = []
    for config in configs:
        print(f"Starting {config['id']}: {config['name']}")
        created = client.ok("POST", "/api/v1/admin/configs",
                            {"kind": "RETRIEVAL", "name": config["name"], "payload": config["payload"]},
                            client.headers())
        cid = created["id"]
        client.ok("POST", f"/api/v1/admin/configs/{cid}/publish", {}, client.headers())
        request = {"datasetVersionId": dataset_id, "snapshotId": snapshot_id, "configId": cid,
                   "split": args.split, "repeats": 1, "mode": "RAG", "strict": True}
        estimate = client.ok("POST", "/api/v1/admin/eval/estimates", request, client.headers())
        estimated_cost = estimate.get("estimatedCostCny")
        assert_that(type(estimated_cost) in (float, int)
                    and math.isfinite(estimated_cost) and estimated_cost <= args.max_cost_cny,
                    f"{config['id']}: unknown/excessive estimated cost")
        request.update({"estimateId": estimate["estimateId"], "maxCostCny": args.max_cost_cny})
        headers = client.headers()
        headers["Idempotency-Key"] = f"ablation-{execution_id}-{config['id']}"
        started = client.ok("POST", "/api/v1/admin/eval/runs", request, headers)
        run_id = started["runId"]
        deadline = time.monotonic() + args.timeout_seconds
        while True:
            time.sleep(2)
            result = client.ok("GET", f"/api/v1/admin/eval/runs/{run_id}")
            assert_that(time.monotonic() < deadline, f"{config['id']}: timed out")
            if result["status"] not in ("QUEUED", "RUNNING"):
                break
        exported = client.ok("POST", f"/api/v1/admin/eval/runs/{run_id}/exports",
                             {"mode": "REDACTED"}, client.headers())
        runs.append({"experimentId": config["id"], "configId": cid, "runId": run_id,
                     "status": result["status"], "estimatedCostCny": estimated_cost,
                     "metrics": result.get("metrics") or {}, "redactedExport": exported})
    comparison = client.ok("GET", "/api/v1/admin/eval/comparisons?runIds=" +
                           local_quote(",".join(str(r["runId"]) for r in runs)))
    baseline = runs[0]["metrics"]
    summary = []
    for run in runs:
        m = run["metrics"]
        item = {k: run[k] for k in ("experimentId", "runId", "status")}
        for k in ("hitAt5", "recallAt5", "mrr", "ndcgAt5", "taskSuccessRate",
                  "knownCostCny", "observedP95Ms"):
            item[k] = m.get(k)
        for key, output in (("recallAt5", "deltaRecallAt5FromB0"),
                            ("ndcgAt5", "deltaNdcgAt5FromB0")):
            item[output] = m[key] - baseline[key] if m.get(key) is not None and baseline.get(key) is not None else None
        summary.append(item)
    eligible = [x for x in summary if x["status"] == "COMPLETED" and x["ndcgAt5"] is not None]
    selected = sorted(eligible, key=lambda x: (-x["ndcgAt5"], -(x["recallAt5"] or 0), x["knownCostCny"] or 0))
    report = {
        "schemaVersion": 1, "executionId": execution_id, "startedAt": started_at,
        "finishedAt": datetime.now(timezone.utc).isoformat(), "datasetVersionId": dataset_id,
        "snapshotId": snapshot_id, "split": args.split, "mode": "RAG",
        "comparable": bool(comparison.get("comparable")), "comparisonReason": comparison.get("reason", ""),
        "selectionRule": "highest ndcgAt5, then recallAt5, then lower knownCostCny",
        "selectedExperimentId": selected[0]["experimentId"] if selected else None,
        "summary": summary, "runs": runs,
    }
    out = ROOT / ".runtime" / "reports" / f"retrieval-ablation-{execution_id}.json"
    report_json(out, report)
    assert_that(report["comparable"], f"Runs not comparable. Report: {out}")
    assert_that(all(r["status"] == "COMPLETED" for r in runs), f"Incomplete runs. Report: {out}")
    print(f"Ablation completed. Selected {report['selectedExperimentId']}. Report: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    val = sub.add_parser("validate")
    val.add_argument("path", type=Path)
    val.add_argument("--expected-count", type=int, default=0)
    val.add_argument("--source-root", type=Path)
    val.add_argument("--require-runtime-bindings", action="store_true")
    bind_parser = sub.add_parser("bind")
    bind_parser.add_argument("path", type=Path)
    bind_parser.add_argument("--binding-path", type=Path, default=ROOT / ".runtime" / "evaluation-bindings.json")
    bind_parser.add_argument("--output-path", type=Path, default=ROOT / ".runtime" / "eval-bound.jsonl")
    abl = sub.add_parser("ablation")
    abl.add_argument("--base-url", default="http://127.0.0.1:8080")
    abl.add_argument("--dataset-version-id", default="")
    abl.add_argument("--snapshot-id", default="")
    abl.add_argument("--dataset-path", type=Path)
    abl.add_argument("--dataset-name", default="enterprise-golden-dataset")
    abl.add_argument("--approved-synthetic-or-redacted", action="store_true")
    abl.add_argument("--split", choices=("DEV", "TEST"), default="TEST")
    abl.add_argument("--max-cost-cny", type=float, default=10)
    abl.add_argument("--timeout-seconds", type=int, default=7200)
    abl.add_argument("--config-path", type=Path, default=ROOT / "experiments" / "retrieval-ablation.json")
    abl.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            print(json.dumps(validate(args.path, args.expected_count, args.source_root,
                                      args.require_runtime_bindings), ensure_ascii=False, indent=2))
        elif args.action == "bind":
            bind(args.path, args.binding_path, args.output_path)
        else:
            ablation(args)
    except (CheckError, ValueError, KeyError, OSError) as exc:
        print(f"Evaluation error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
