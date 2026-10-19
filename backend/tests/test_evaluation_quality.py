import json
from datetime import datetime, timezone

import pytest
import importlib.util
from pathlib import Path

from app.db import get_db_connection
from app.errors import ApiException
from app.evaluation.judge import (
    parse_verdicts,
    parse_full_answer_verdicts,
    judge_answer,
    RUBRIC_VERSION,
    INPUT_CONTRACT_VERSION,
    EVALUATION_SCOPE,
)
from app.evaluation.runner import EvaluationRunner
from app.evaluation.metrics import EvaluationMetrics
from app.qa.model_gateway import ModelGateway
from app.identity.deps import Actor
from app.models import eval_result_table, eval_run_table


@pytest.mark.parametrize("fact,claim,label", [
    ("COVERED", "SUPPORTED", "CORRECT"),
    ("COVERED", "UNSUPPORTED", "WRONG"),
    ("CONTRADICTED", "SUPPORTED", "WRONG"),
    ("MISSING", "SUPPORTED", "WRONG"),
    ("COVERED", "UNCERTAIN", "UNDETERMINED"),
])
def test_semantic_scoring_requires_facts_and_support(fact, claim, label):
    raw = json.dumps({"facts": [{"index": 0, "verdict": fact}],
                      "claims": [{"index": 0, "verdict": claim}]})
    assert parse_verdicts(raw, ["gold"], ["answer"])["correctness"] == label


def test_incomplete_or_duplicate_judgments_fail_closed():
    with pytest.raises(ValueError):
        parse_verdicts('{"facts":[],"claims":[]}', ["gold"], ["answer"])
    with pytest.raises(ValueError):
        parse_verdicts(json.dumps({"facts": [{"index": 0, "verdict": "COVERED"}] * 2,
                                  "claims": []}), ["a", "b"], [])


def test_review_changes_summary_without_erasing_automatic_result():
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-admin-1", admin=True)
    now = datetime.now(timezone.utc)
    with get_db_connection() as conn:
        for i in range(2):
            conn.execute(eval_result_table.insert().values(
                tenant_id=actor.tenant_id, id=f"result-{i}", eval_run_id="run-quality",
                dataset_version_id="dataset", case_id=f"case-{i}", repeat_no=0,
                status="COMPLETED", metrics_json={"taskSuccess": None, "costCny": None},
                created_at=now, updated_at=now,
            ))
        conn.commit()
    before = EvaluationRunner.aggregate(actor, "run-quality")
    assert before["taskSuccessRate"] is None
    assert before["totalCostCny"] is None
    EvaluationRunner.review(actor, "result-0", {"labels": {"correctness": "CORRECT"}, "reason": "verified"})
    after = EvaluationRunner.aggregate(actor, "run-quality")
    assert after["taskSuccessRate"] == 1.0
    assert after["judgmentCoverage"] == 0.5
    assert after["unknownCostCases"] == 2
    EvaluationRunner.review(actor, "result-0", {"labels": {"correctness": "WRONG"}, "reason": "corrected"})
    assert EvaluationRunner.aggregate(actor, "run-quality")["taskSuccessRate"] == 0.0


def test_compare_rejects_different_modes(monkeypatch):
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-admin-1", admin=True)
    monkeypatch.setattr(EvaluationRunner, "get_run", lambda actor, rid: {"manifest": {
        "datasetVersionId": "d", "snapshotId": "s", "split": "TEST", "mode": rid,
    }})
    monkeypatch.setattr(EvaluationRunner, "results", lambda *args: [])
    comparison = EvaluationRunner.compare(actor, ["RAG", "AGENT"])
    assert comparison["comparable"] is False
    assert comparison["differingFields"] == ["mode"]


def test_matching_quote_from_wrong_version_or_unknown_document_is_not_a_hit():
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-admin-1", admin=True)
    case = {"expected": {"evidenceGroups": [{"fact": "f", "alternatives": [{
        "documentId": "doc", "versionId": "v1", "quote": "known fact", "relevanceGrade": 3,
    }]}]}}
    for hit in [{"documentId": "doc", "versionId": "v2", "text": "known fact"},
                {"versionId": "v1", "text": "known fact"}]:
        assert EvaluationRunner._compute_retrieval_metrics(actor, case, [hit], [hit])["hit"] == 0
    hit = {"documentId": "doc", "versionId": "v1", "text": "known fact"}
    assert EvaluationRunner._compute_retrieval_metrics(actor, case, [hit], [hit])["hit"] == 1


def test_release_gate_rejects_unknowns_and_retrieval_only():
    spec = importlib.util.spec_from_file_location("gate", Path(__file__).parents[2] / "scripts/check_evaluation_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    report = {"manifest": {"mode": "RAG"}, "metrics": {
        "planned": 10, "completed": 10, "judgmentCoverage": 1,
        "taskSuccessRate": 0.9, "totalCostCny": 1.5,
    }}
    assert gate.failures(report, 0.9, 2) == []
    report["metrics"]["totalCostCny"] = None
    assert gate.failures(report, 0.9, 2)
    report["metrics"]["totalCostCny"] = 1.5
    report["manifest"]["mode"] = "RETRIEVAL_ONLY"
    assert gate.failures(report, 0.9, 2)


def _v2_payload(fact="COVERED", claim="SUPPORTED", relevance="ON_TOPIC",
                alignment="ALIGNED", expectation="SATISFIED"):
    return json.dumps({
        "facts": [{"index": 0, "verdict": fact}],
        "claims": [{"index": 0, "verdict": claim}],
        "answerRelevance": {"verdict": relevance, "reason": "relevance reason"},
        "claimAlignment": {"verdict": alignment, "reason": "alignment reason"},
        "expectationAssessment": {"verdict": expectation, "reason": "expectation reason"},
        "reason": "overall",
    })


def test_v2_relevance_is_observe_only_for_correctness():
    result = parse_full_answer_verdicts(
        _v2_payload(relevance="OFF_TOPIC"),
        ["gold"],
        [{"text": "claim"}],
        "ANSWER",
    )
    assert result["correctness"] == "CORRECT"
    assert result["answerRelevance"]["verdict"] == "OFF_TOPIC"
    assert result["rubricVersion"] == RUBRIC_VERSION


def test_v2_claim_alignment_mismatch_fails_closed_without_inventing_wrongness():
    result = parse_full_answer_verdicts(
        _v2_payload(alignment="MISMATCH"),
        ["gold"],
        [{"text": "claim"}],
        "ANSWER",
    )
    assert result["correctness"] == "UNDETERMINED"
    assert result["claimAlignment"]["verdict"] == "MISMATCH"


def test_v2_explicit_failure_wins_over_alignment_uncertainty():
    result = parse_full_answer_verdicts(
        _v2_payload(fact="CONTRADICTED", alignment="MISMATCH"),
        ["gold"],
        [{"text": "claim"}],
        "ANSWER",
    )
    assert result["correctness"] == "WRONG"


def test_v2_abstain_can_be_correct_without_claims_or_faithfulness():
    raw = json.dumps({
        "facts": [],
        "claims": [],
        "answerRelevance": {"verdict": "ON_TOPIC", "reason": "explains refusal"},
        "claimAlignment": {"verdict": "ALIGNED", "reason": "no factual claims"},
        "expectationAssessment": {"verdict": "SATISFIED", "reason": "matches gold abstain reason"},
        "reason": "valid abstention",
    })
    result = parse_full_answer_verdicts(
        raw, [], [], "ABSTAIN", {"status": "ABSTAIN", "abstainReason": "资料不足，无法可靠回答"}
    )
    assert result["correctness"] == "CORRECT"
    assert result["faithfulness"] is None


def test_v2_abstain_without_gold_reason_stays_undetermined():
    raw = json.dumps({
        "facts": [],
        "claims": [],
        "answerRelevance": {"verdict": "ON_TOPIC", "reason": "relevant refusal"},
        "claimAlignment": {"verdict": "ALIGNED", "reason": "no factual claims"},
        "expectationAssessment": {"verdict": "SATISFIED", "reason": "judge guessed this was a good refusal"},
        "reason": "looks fine",
    })
    result = parse_full_answer_verdicts(
        raw, [], [], "ABSTAIN", {"status": "ABSTAIN"}
    )
    assert result["correctness"] == "UNDETERMINED"


def test_history_context_revalidates_every_run_before_reading(monkeypatch):
    checked = []

    def deny(actor, run_id):
        checked.append(run_id)
        raise ApiException(409, "SOURCE_CHANGED", "revoked history source")

    monkeypatch.setattr("app.qa.context.assert_run_sources_readable", deny)
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-operator-1", admin=False)

    with pytest.raises(ApiException) as exc:
        EvaluationRunner._judge_conversation_context(actor, ["history-run-1", "final-run-2"])
    assert exc.value.code == "SOURCE_CHANGED"
    assert checked == ["history-run-1"]


def test_full_answer_judge_sends_real_answer_history_and_contract(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return _v2_payload()

    monkeypatch.setattr(ModelGateway, "chat", fake_chat)
    result = judge_answer(
        "tenant",
        "run",
        "follow-up question",
        {"status": "ANSWER", "factPoints": ["gold"]},
        {
            "answer": "the exact user-visible answer",
            "claims": [{"text": "claim"}],
            "evidence": [{"evidenceId": "E1", "text": "source"}],
        },
        answer_text="the exact user-visible answer",
        answer_present=True,
        conversation_context=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "follow-up question"},
        ],
    )
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["answerText"] == "the exact user-visible answer"
    assert payload["conversationContext"][1]["content"] == "earlier answer"
    assert payload["inputContractVersion"] == INPUT_CONTRACT_VERSION
    assert payload["evaluationScope"] == EVALUATION_SCOPE
    assert result["correctness"] == "CORRECT"


def test_missing_and_blank_answers_are_distinct_and_do_not_call_model(monkeypatch):
    def should_not_call(**kwargs):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(ModelGateway, "chat", should_not_call)
    missing = judge_answer(
        "tenant", "run", "q",
        {"status": "ANSWER", "factPoints": ["gold"]},
        {"claims": [], "evidence": []},
        answer_present=False,
    )
    blank = judge_answer(
        "tenant", "run", "q",
        {"status": "ANSWER", "factPoints": ["gold"]},
        {"answer": "   ", "claims": [], "evidence": []},
        answer_text="   ",
        answer_present=True,
    )
    assert missing["correctness"] == "UNDETERMINED"
    assert missing["inputFailureReason"] == "MISSING_ANSWER_TEXT"
    assert blank["correctness"] == "WRONG"
    assert blank["inputFailureReason"] == "BLANK_ANSWER"


def test_calibration_formulas_exclude_undetermined_pairs():
    correctness = EvaluationMetrics.correctness_calibration([
        ("CORRECT", "WRONG"),
        ("WRONG", "CORRECT"),
        ("CORRECT", "CORRECT"),
        ("UNDETERMINED", "CORRECT"),
    ])
    assert correctness["compared"] == 3
    assert correctness["falsePositiveRate"] == 1.0
    assert correctness["falseNegativeRate"] == 0.5
    assert correctness["agreementRate"] == 1.0 / 3.0

    relevance = EvaluationMetrics.categorical_comparison(
        [("ON_TOPIC", "ON_TOPIC"), ("OFF_TOPIC", "PARTLY_OFF_TOPIC")],
        ["ON_TOPIC", "PARTLY_OFF_TOPIC", "OFF_TOPIC", "UNCERTAIN"],
    )
    assert relevance["compared"] == 2
    assert relevance["agreementRate"] == 0.5
    assert relevance["confusionMatrix"]["PARTLY_OFF_TOPIC"]["OFF_TOPIC"] == 1


def test_review_accepts_partial_dimension_labels():
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-admin-1", admin=True)
    now = datetime.now(timezone.utc)
    result_id = "result-relevance-review"
    with get_db_connection() as conn:
        conn.execute(eval_result_table.insert().values(
            tenant_id=actor.tenant_id, id=result_id, eval_run_id="run-relevance-review",
            dataset_version_id="dataset", case_id="case", repeat_no=0,
            status="COMPLETED", metrics_json={"taskSuccess": None, "costCny": None},
            judge_json={"correctness": "UNDETERMINED",
                        "answerRelevance": {"verdict": "ON_TOPIC", "reason": "auto"}},
            created_at=now, updated_at=now,
        ))
        conn.commit()
    saved = EvaluationRunner.review(actor, result_id, {
        "labels": {"answerRelevance": "OFF_TOPIC"},
        "reason": "human review",
    })
    assert saved["labels"] == {"answerRelevance": "OFF_TOPIC"}


def test_planned_success_denominator_separates_execution_failure_from_semantic_unknown():
    actor = Actor(tenant_id="tenant-demo-1", user_id="user-admin-1", admin=True)
    now = datetime.now(timezone.utc)
    run_id = "run-v2-denominator"
    with get_db_connection() as conn:
        conn.execute(eval_run_table.insert().values(
            tenant_id=actor.tenant_id,
            id=run_id,
            dataset_version_id="dataset",
            snapshot_id="snapshot",
            config_id="config",
            job_id="job",
            status="COMPLETED",
            manifest_json={
                "mode": "RAG",
                "rubricVersion": RUBRIC_VERSION,
                "inputContractVersion": INPUT_CONTRACT_VERSION,
                "evaluationScope": EVALUATION_SCOPE,
                "relevancePolicy": "OBSERVE_ONLY",
            },
            metrics_json=None,
            created_at=now,
            updated_at=now,
        ))
        rows = [
            ("result-v2-success", "COMPLETED", {"taskSuccess": True, "costCny": 0.1},
             {"correctness": "CORRECT", "answerRelevance": {"verdict": "ON_TOPIC", "reason": "ok"}}),
            ("result-v2-unknown", "COMPLETED", {"taskSuccess": None, "costCny": 0.1},
             {"correctness": "UNDETERMINED", "answerRelevance": {"verdict": "UNCERTAIN", "reason": "unknown"}}),
            # 历史错误路径会持久化 taskSuccess=False；该执行失败不能进入语义正确性分母，否则会污染质量指标。
            ("result-v2-failed", "FAILED", {"taskSuccess": False, "costCny": 0.1},
             {"correctness": "WRONG"}),
        ]
        for i, (rid, status, metrics, judge) in enumerate(rows):
            conn.execute(eval_result_table.insert().values(
                tenant_id=actor.tenant_id,
                id=rid,
                eval_run_id=run_id,
                dataset_version_id="dataset",
                case_id=f"case-v2-{i}",
                repeat_no=0,
                status=status,
                metrics_json=metrics,
                judge_json=judge,
                error_code="QA_EXECUTION_FAILED" if status == "FAILED" else None,
                created_at=now,
                updated_at=now,
            ))
        conn.commit()

    metrics = EvaluationRunner.aggregate(actor, run_id)
    assert metrics["planned"] == 3
    assert metrics["successful"] == 1
    assert metrics["judgedCases"] == 1
    assert metrics["unknown"] == 2
    assert metrics["executionFailed"] == 1
    assert metrics["taskSuccessRate"] == 1.0
    assert metrics["plannedTaskSuccessRate"] == 1.0 / 3.0
    assert metrics["judgmentCoverage"] == 1.0 / 3.0
    assert metrics["answerRelevance"]["counts"]["ON_TOPIC"] == 1
    assert metrics["answerRelevance"]["counts"]["UNCERTAIN"] == 1
    assert metrics["answerRelevance"]["unevaluated"] == 1
