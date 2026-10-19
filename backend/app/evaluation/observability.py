"""Versioned evaluation samples and deterministic diagnostic metrics.

These fields never replace the existing rubric, aggregate, or quality gate.
"""
import hashlib
from collections import Counter

CONTRACT_VERSION = "eobs-v1"
SCOPE = "FINAL_TURN_WITH_HISTORY"


def metric(name, *, scope, status, numerator=None, denominator=None,
           value=None, error_code=None, method="DETERMINISTIC", **details):
    return {"name": name, "version": CONTRACT_VERSION, "scope": scope,
            "method": method, "requiredInputs": details.pop("requiredInputs", []),
            "status": status, "value": value if status == "OK" else None,
            "numerator": numerator, "denominator": denominator,
            "errorCode": error_code, **details}


def build_sample(case, manifest, result_id, repeat_no, output=None, history=None, diagnostics=None):
    output = output if isinstance(output, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    evidence = output.get("evidence") or []
    actual = [d for d in diagnostics if d.get("stage") == "MODEL_CONTEXT" and d.get("purpose") != "JUDGE"]
    return {"contractVersion": CONTRACT_VERSION, "sampleId": result_id,
            "caseKey": case.get("caseKey"), "repeatNo": repeat_no,
            "evaluationScope": SCOPE, "mode": manifest.get("mode"),
            "question": output.get("question") or (case.get("question") or {}).get("text"),
            "history": {"status": "KNOWN" if history is not None else "UNKNOWN",
                        "messages": history if history is not None else None},
            "answer": {"present": "answer" in output, "text": output.get("answer"),
                       "terminalStatus": output.get("status")},
            "claims": output.get("claims"), "expected": case.get("expected"),
            "persona": case.get("persona"), "snapshotId": manifest.get("snapshotId"),
            "configId": manifest.get("configId"),
            "sourceKinds": {"documents": sum(e.get("sourceType", "DOCUMENT") != "TOOL" for e in evidence),
                            "tools": sum(e.get("sourceType") == "TOOL" for e in evidence),
                            "mockTools": sum(e.get("sourceType") == "TOOL" and e.get("mock") is True for e in evidence)},
            "finalEvidence": evidence,
            "actualContext": {"status": "CAPTURED_PARTIAL" if actual else "UNKNOWN",
                              "complete": False, "historyMapping": "UNKNOWN",
                              "calls": actual},
            "contentStatus": "KNOWN" if "answer" in output else "UNKNOWN"}


def describe_metrics(case, result_status, values, judge):
    groups = (case.get("expected") or {}).get("evidenceGroups") or []
    descriptors = []
    for name in ("hit", "recall", "mrr", "ndcg", "factCoverage", "faithfulness"):
        value = values.get(name, judge.get(name))
        required = ["goldFactGroups", "rankedCanonicalChunks"] if name in {"hit", "recall", "mrr", "ndcg"} else ["fullAnswer", "judgeVerdicts"]
        denominator = len(groups) if name == "recall" else None
        numerator = round(value * denominator) if type(value) in {int, float} and denominator else None
        status, error = ("OK", None) if type(value) in {int, float} else ("UNDETERMINED", "MISSING_INPUT")
        if name in {"hit", "recall", "mrr", "ndcg"} and not groups:
            status, error = "NOT_APPLICABLE", "EMPTY_GOLD"
        if name in {"factCoverage", "faithfulness"}:
            if judge.get("method") in {"STATUS_ONLY_V2", None}:
                error = "NO_SEMANTIC_JUDGMENT"
            verdicts = (judge.get("verdicts") or {}).get("facts" if name == "factCoverage" else "claims")
            if isinstance(verdicts, list):
                denominator = len(verdicts)
                numerator = sum(v.get("verdict") == ("COVERED" if name == "factCoverage" else "SUPPORTED") for v in verdicts)
                if not verdicts:
                    status, error = "NOT_APPLICABLE", "EMPTY_ASSERTIONS"
                elif any(v.get("verdict") == "UNCERTAIN" for v in verdicts):
                    status, error = "UNDETERMINED", "UNCERTAIN_VERDICTS"
        if result_status == "FAILED":
            status, error = "ERROR", "EVALUATION_FAILED"
        descriptors.append(metric(name, scope=SCOPE, status=status, numerator=numerator,
            denominator=denominator, value=value, error_code=error, requiredInputs=required,
            method=(judge.get("method") or "UNKNOWN") if name in {"factCoverage", "faithfulness"} else "DETERMINISTIC",
            rubricVersion=judge.get("rubricVersion") if name in {"factCoverage", "faithfulness"} else None))
    return descriptors


def stage_coverage(expected, snapshot, chunks, *, k=None):
    """Fact-group coverage on canonical candidates or exact sent document ranges.

    chunks is keyed by (build_id, chunk_id), with canonical text and version IDs.
    Missing/ambiguous provenance invalidates the metric, never turns into zero.
    """
    actual = snapshot.get("stage") == "MODEL_CONTEXT"
    if k is None and not actual:
        k = snapshot.get("k")
    scope = "SENT_STRUCTURED_DOCUMENTS" if actual else (
        "RETRIEVAL_CONTEXT" if snapshot.get("stage") in {"parent_expansion", "context_budget", "finalize_hits"}
        else "CANONICAL_CANDIDATES")
    groups = expected.get("evidenceGroups") or []
    base = {"stage": snapshot.get("stage"), "retrievalIndex": snapshot.get("retrievalIndex"),
            "routeIndex": snapshot.get("routeIndex"), "callId": snapshot.get("callId"), "k": k,
            "requiredInputs": ["goldFactGroups", "canonicalVersions", "exactSentRanges" if actual else "orderedStageSnapshot"]}
    if not groups:
        return metric("stageFactRecall", scope=scope, status="NOT_APPLICABLE", denominator=0,
                      error_code="EMPTY_GOLD", **base)
    if any(not g.get("alternatives") or any(not all(a.get(key) for key in
        ("documentId", "versionId", "quote")) for a in g['alternatives']) for g in groups):
        return metric("stageFactRecall", scope=scope, status="UNDETERMINED", error_code="INVALID_GOLD", **base)
    complete = snapshot.get("sourceRangesComplete") if actual else snapshot.get("complete")
    if complete is not True:
        return metric("stageFactRecall", scope=scope, status="UNDETERMINED", error_code="INCOMPLETE_SNAPSHOT", **base)
    items = snapshot.get("items") or []
    if k is not None:
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        items = items[:k]
    retained = []
    for item in items:
        if item.get("sourceType") == "TOOL":
            continue
        build_id = item.get("buildId")
        chunk_id = item.get("contextChunkId") if actual else item.get("chunkId")
        row = chunks.get((build_id, chunk_id))
        if not actual and snapshot.get("stage") in {"parent_expansion", "context_budget", "finalize_hits"}:
            # Hashes remain local. A child fallback must not be mistaken for its parent.
            matching = [chunks.get((build_id, cid)) for cid in
                        dict.fromkeys((item.get("parentChunkId"), item.get("chunkId")))]
            matching = [r for r in matching if r and item.get("textSha256") and
                        hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == item["textSha256"]]
            row = matching[0] if matching else None
        if not row or row.get("documentId") != item.get("documentId"):
            return metric("stageFactRecall", scope=scope, status="UNDETERMINED", error_code="SOURCE_UNRESOLVED", **base)
        text = row["text"]
        if actual:
            start, end = item.get("contextStart"), item.get("contextEnd")
            if item.get("mappingStatus") != "EXACT" or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                return metric("stageFactRecall", scope=scope, status="UNDETERMINED", error_code="UNKNOWN_SENT_RANGE", **base)
            text = text[start:end]
        retained.append((row["documentId"], row["versionId"], text))
    found = {index for index, group in enumerate(groups) if any(
        alt.get("quote") and alt.get("documentId") == doc and alt.get("versionId") == version
        and alt["quote"] in text for alt in group["alternatives"] for doc, version, text in retained)}
    return metric("stageFactRecall", scope=scope, status="OK", numerator=len(found),
                  denominator=len(groups), value=len(found)/len(groups), **base)


def metric_coverage(rows):
    """Count legacy/missing, unknown and error observations without zero filling."""
    keys = {(m["name"], m["scope"], m.get("stage")) for row in rows
            for m in (row.get("metricDiagnostics") or []) + (row.get("stageCoverage") or [])}
    result = []
    for name, scope, stage in sorted(keys, key=str):
        counts = Counter()
        values = []
        for row in rows:
            entries = [m for m in (row.get("metricDiagnostics") or []) + (row.get("stageCoverage") or [])
                       if (m["name"], m["scope"], m.get("stage")) == (name, scope, stage)]
            for entry in entries or [{"status": "UNDETERMINED"}]:
                counts[entry["status"]] += 1
                if entry["status"] == "OK" and type(entry.get("value")) in {int, float}:
                    values.append(entry["value"])
        result.append({"name": name, "version": CONTRACT_VERSION, "scope": scope, "stage": stage,
                       "plannedCases": len(rows), "observations": sum(counts.values()),
                       "computed": counts["OK"], "notApplicable": counts["NOT_APPLICABLE"],
                       "unknown": counts["UNDETERMINED"], "errors": counts["ERROR"],
                       "mean": sum(values)/len(values) if values else None})
    return result
