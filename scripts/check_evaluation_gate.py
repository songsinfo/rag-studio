"""Check an exported evaluation report; unknown/incomplete results fail closed."""
import argparse
import json
import math
from pathlib import Path


def failures(report, minimum_success, maximum_cost):
    metrics = report.get("metrics", {})
    reasons = []
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not metrics.get("planned") or metrics.get("completed") != metrics.get("planned"):
        reasons.append("evaluation incomplete or failed")
    coverage = metrics.get("judgmentCoverage")
    if not finite(coverage) or coverage != 1:
        reasons.append("not all cases have a quality judgment")
    success = metrics.get("taskSuccessRate")
    if not finite(success) or success < minimum_success:
        reasons.append("quality success rate below threshold or unknown")
    cost = metrics.get("totalCostCny")
    if not finite(cost) or cost < 0 or cost > maximum_cost:
        reasons.append("total cost unknown or exceeds limit")
    if report.get("manifest", {}).get("mode") not in {"RAG", "AGENT"}:
        reasons.append("retrieval-only report cannot approve answer quality")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-success", type=float, required=True)
    parser.add_argument("--max-cost", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.min_success <= 1 or not math.isfinite(args.max_cost) or args.max_cost < 0:
        parser.error("invalid quality or cost threshold")
    reasons = failures(json.loads(args.report.read_text(encoding="utf-8-sig")), args.min_success, args.max_cost)
    print(json.dumps({"passed": not reasons, "reasons": reasons}, ensure_ascii=False))
    raise SystemExit(1 if reasons else 0)


if __name__ == "__main__":
    main()
