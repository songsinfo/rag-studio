import json

import pytest
from sqlalchemy import select, update

from app.db import get_db_connection
from app.errors import ApiException
from app.evaluation.assertion_diagnostic import (
    parse_assertions, parse_support, diagnose_answer, start_diagnostic,
    get_diagnostic, calibrate_diagnostic, calibration_summary,
)
from app.evaluation.catalog import EvaluationCatalog
from app.evaluation.runner import EvaluationRunner
from app.models import eval_case_table, eval_result_table, eval_run_table, config_revision_table, model_call_table, app_user_table
from app.qa.model_gateway import ModelGateway
from app.qa.model_usage import invoke_tracked
from app.qa.service import QaService
from app.runtime.worker import DurableJobWorker
from test_evaluation import eval_env


def test_independent_extraction_does_not_receive_generated_claims(monkeypatch):
    payloads = []
    replies = iter(['{"assertions":[{"text":"错断言"}]}',
                    '{"verdicts":[{"index":0,"support":"UNSUPPORTED","represented":"MISSING"}]}'])
    def chat(*args, **kwargs):
        payloads.append(json.loads(kwargs['messages'][-1]['content']))
        return next(replies)
    monkeypatch.setattr(ModelGateway, 'chat', chat)
    checks = []
    result = diagnose_answer('tenant', 'job', {'answer': '错断言', 'claims': ['私有生成claims'], 'evidence': []},
                             '问题', [], lambda: checks.append(True))
    assert 'claims' not in payloads[0] and '私有生成claims' not in str(payloads[0])
    assert payloads[1]['claims'] == ['私有生成claims'] and len(checks) == 2
    assert result['faithfulness'] == 0 and result['missingFromGeneratedClaims'] == [0]
    assert result['policy'] == 'OBSERVE_ONLY'


def test_assertion_parse_rejects_transformed_ambiguous_and_duplicate_ranges():
    assert parse_assertions('{"assertions":[{"text":"证据"}]}', '这是证据')[0]['start'] == 2
    for raw, answer in [('证据', '证据证据'), ('变换', '证据')]:
        with pytest.raises(ValueError):
            parse_assertions(json.dumps({'assertions': [{'text': raw}]}), answer)
    with pytest.raises(ValueError):
        parse_assertions('{"assertions":[{"text":"a","start":0,"end":1},{"text":"a","start":0,"end":1}]}', 'a')
    with pytest.raises(ValueError):
        parse_support('{"verdicts":[]}', [{'text': 'a'}])


def test_unknown_empty_and_human_denominators_do_not_invent_perfect_scores():
    assert parse_support('{"verdicts":[]}', [])['faithfulness'] is None
    result = parse_support('{"verdicts":[{"index":0,"support":"UNCERTAIN","represented":"UNCERTAIN"}]}', [{'text': 'a'}])
    assert result['status'] == 'UNDETERMINED' and result['faithfulness'] is None
    summary = calibration_summary([{'resultId': 'r', 'verdicts': [
        {'index': 0, 'support': 'SUPPORTED'}, {'index': 1, 'support': 'UNCERTAIN'}]}],
        [{'resultId': 'r', 'assertionIndex': 0, 'support': 'UNSUPPORTED'},
         {'resultId': 'r', 'assertionIndex': 1, 'support': 'SUPPORTED'}])
    assert summary['agreement'] == 0 and summary['falseSupportedRate'] == 1
    assert summary['compared'] == summary['unknown'] == 1


@pytest.fixture
def completed_result(eval_env):
    actor = eval_env['admin']
    with get_db_connection() as conn:
        for alias in ('chat-main', 'judge-main'):
            conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
                id='cfg-model-'+alias, kind='MODEL', name='synthetic pricing', revision=1,
                sha256=b'0'*32, status='PUBLISHED', payload={'modelAlias': alias, 'currency': 'CNY',
                    'inputPricePerMillion': 1, 'outputPricePerMillion': 1}))
        conn.commit()
    snapshot = EvaluationCatalog.create_snapshot(actor, {'name': 'Synthetic diagnostics',
        'documentIds': [eval_env['doc_id']], 'approvedSyntheticOrRedacted': True})
    conv = QaService.create_conversation(actor, {'title': 'synthetic'})
    qa_id = QaService.submit_run(actor, conv['conversationId'],
        {'question': '知识库系统采用什么分块策略？', 'segmentId': conv['currentSegmentId']},
        idempotency_key='fixture', evaluation_scope={**snapshot['manifest'], 'snapshotId': snapshot['id']})['runId']
    dataset = EvaluationCatalog.create_dataset(actor, {'name': 'Synthetic diagnostic Gold'})
    EvaluationCatalog.save_cases(actor, dataset['id'], [{'caseKey': 'synthetic', 'split': 'TEST',
        'groupKey': 'g', 'category': 'FACT', 'question': {'text': '合成问题'},
        'persona': {'loginName': 'eval_admin'}, 'expected': {'status': 'ANSWER', 'evidenceGroups': []},
        'annotationStatus': 'HUMAN_REVIEWED'}])
    with get_db_connection() as conn:
        case_id = conn.execute(select(eval_case_table.c.id).where(eval_case_table.c.dataset_version_id == dataset['id'])).scalar()
        conn.execute(eval_run_table.insert().values(tenant_id=actor.tenant_id, id='original-eval',
            dataset_version_id=dataset['id'], snapshot_id=snapshot['id'], config_id='cfg-enterprise-retrieval-v2',
            job_id='original-job', status='COMPLETED', manifest_json={'snapshotId': snapshot['id'], 'mode': 'AGENT'}))
        conn.execute(eval_result_table.insert().values(tenant_id=actor.tenant_id, id='original-result',
            eval_run_id='original-eval', dataset_version_id=dataset['id'], case_id=case_id,
            qa_run_id=qa_id, status='COMPLETED', metrics_json={'taskSuccess': None},
            judge_json={'correctness': 'UNDETERMINED', 'qaRunIds': [qa_id]}))
        conn.commit()
    return actor, case_id, qa_id


def test_explicit_worker_job_budget_calibration_and_authoritative_results(monkeypatch, completed_result):
    actor, _, qa_id = completed_result
    answer = QaService.get_run(actor, qa_id)['answer']
    replies = iter([json.dumps({'assertions': [{'text': answer}]}),
                    '{"verdicts":[{"index":0,"support":"SUPPORTED","represented":"PRESENT"}]}'])
    def tracked_chat(tenant_id, run_id, purpose, **kwargs):
        return invoke_tracked(lambda: next(replies), tenant_id=tenant_id, job_id=kwargs['job_id'],
            purpose=purpose, alias='judge-main', payload={'messages': kwargs['messages']}, max_output_tokens=20,
            usage_reader=lambda _: {'input_tokens': 10, 'output_tokens': 5})
    monkeypatch.setattr(ModelGateway, 'chat', tracked_chat)
    request = {'resultIds': ['original-result'], 'maxCostCny': .1}
    first = start_diagnostic(actor, 'original-eval', request, 'diagnostic-fixture')
    assert first == start_diagnostic(actor, 'original-eval', request, 'diagnostic-fixture')
    assert DurableJobWorker(kinds={'EVALUATE'}).run_once()
    detail = get_diagnostic(actor, first['jobId'])
    assert detail['status'] == 'COMPLETED' and detail['modelCalls'] == 2
    assert detail['results'][0]['faithfulness'] == 1 and detail['costCny'] > 0
    assert detail['calibration']['status'] == 'NOT_CALIBRATED'
    labels = {'labels': [{'resultId': 'original-result', 'assertionIndex': 0, 'support': 'UNSUPPORTED'}]}
    summary = calibrate_diagnostic(actor, first['jobId'], labels)
    assert summary['falseSupportedRate'] == 1
    labels['labels'][0]['support'] = 'SUPPORTED'
    calibrate_diagnostic(actor, first['jobId'], labels)
    assert len(get_diagnostic(actor, first['jobId'])['calibrations']) == 2
    with get_db_connection() as conn:
        original = conn.execute(select(eval_result_table).where(eval_result_table.c.id == 'original-result')).one()
        assert original.metrics_json == {'taskSuccess': None} and original.judge_json['correctness'] == 'UNDETERMINED'
        assert conn.execute(select(model_call_table.c.id).where(model_call_table.c.job_id == first['jobId'])).all()
    assert EvaluationRunner.get_run(actor, 'original-eval')['status'] == 'COMPLETED'


def test_unreviewed_gold_and_missing_budget_make_no_model_calls(completed_result):
    actor, case_id, _ = completed_result
    with pytest.raises(ApiException) as missing:
        start_diagnostic(actor, 'original-eval', {'resultIds': ['original-result']}, 'missing-budget')
    assert missing.value.code == 'BUDGET_REQUIRED'
    with get_db_connection() as conn:
        conn.execute(update(eval_case_table).where(eval_case_table.c.id == case_id).values(annotation_status='AUTO_REVIEWED'))
        conn.commit()
    with pytest.raises(ApiException) as unreviewed:
        start_diagnostic(actor, 'original-eval', {'resultIds': ['original-result'], 'maxCostCny': .1}, 'unreviewed')
    assert unreviewed.value.code == 'GOLD_REVIEW_REQUIRED'


def test_new_full_history_sample_is_redacted_when_persona_is_disabled(completed_result):
    actor, _, qa_id = completed_result
    EvaluationRunner._update_result(actor, 'original-result', 'COMPLETED', {'taskSuccess': None},
        {'correctness':'UNDETERMINED','qaRunIds':[qa_id]}, qa_run_id=qa_id)
    before = EvaluationRunner.results(actor, 'original-eval')[0]['judge']['evaluationSample']
    assert before['answer']['present'] and before['history']['status'] == 'KNOWN'
    with get_db_connection() as conn:
        conn.execute(update(app_user_table).where(app_user_table.c.id==actor.user_id).values(status='DISABLED'))
        conn.commit()
    after = EvaluationRunner.results(actor, 'original-eval')[0]['judge']['evaluationSample']
    assert after['contentStatus'] == 'UNKNOWN' and 'answer' not in after and 'history' not in after
