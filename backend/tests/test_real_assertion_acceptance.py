"""Opt-in real independent extraction on user-confirmed synthetic Gold."""
import json
import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import select, update

from app.db import get_db_connection
from app.evaluation.assertion_diagnostic import diagnose_answer, calibration_summary
from app.evaluation.catalog import EvaluationCatalog
from app.models import chunk_table, source_unit_table, config_revision_table, job_table, model_call_table
from test_evaluation import eval_env


@pytest.mark.skipif(os.getenv('RAG_REAL_ASSERTION_ACCEPTANCE') != '1', reason='explicit paid confirmed-Gold experiment only')
def test_real_independent_assertions_on_confirmed_counterexample(monkeypatch, eval_env):
    root = Path(__file__).resolve().parents[2]
    rows = [json.loads(line) for line in (root/'datasets/eobs_assertion_candidates_30.jsonl').read_text(encoding='utf-8').splitlines()]
    case = next(r for r in rows if r['caseKey'] == 'eobs-omitted_error-1')
    assert case['annotationStatus'] == 'HUMAN_REVIEWED' and case['humanLabels'] and case['reviewer']
    config = dotenv_values(root/'.env')
    cap = float(os.environ['RAG_REAL_ASSERTION_MAX_COST_CNY'])
    assert 0 < cap <= 5
    monkeypatch.setenv('LITELLM_BASE_URL', config['LITELLM_BASE_URL'])
    monkeypatch.setenv('LITELLM_MASTER_KEY', config['LITELLM_MASTER_KEY'])
    monkeypatch.setenv('RAG_MONTHLY_BUDGET_CNY', str(cap))
    monkeypatch.setenv('AGENT_CALL_MAX_ATTEMPTS', '1')
    monkeypatch.setenv('AGENT_FALLBACK_MODEL_ALIAS', '')
    actor = eval_env['admin']
    job_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        # Authored counterexample, not an online generated answer or company data.
        conn.execute(update(chunk_table).where(chunk_table.c.tenant_id == actor.tenant_id)
            .values(text_content=case['evidenceText']))
        conn.execute(update(source_unit_table).where(source_unit_table.c.tenant_id == actor.tenant_id)
            .values(text_content=case['evidenceText']))
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id, id='cfg-model-judge-main',
            kind='MODEL', name='conservative acceptance ceiling', revision=1, sha256=b'0'*32, status='PUBLISHED',
            payload={'modelAlias':'judge-main','currency':'CNY','inputPricePerMillion':100,'outputPricePerMillion':200}))
        conn.execute(job_table.insert().values(tenant_id=actor.tenant_id, id=job_id, kind='EVALUATE',
            target_id=job_id, requested_by=actor.user_id, idempotency_key='real-assertion-'+job_id,
            request_hash=b'0'*32, status='RUNNING', checkpoint={'maxCostCny':cap,'maxModelCalls':2}))
        conn.commit()
    snapshot = EvaluationCatalog.create_snapshot(actor, {'name':'Confirmed synthetic counterexample',
        'documentIds':[eval_env['doc_id']], 'approvedSyntheticOrRedacted':True})
    record = {'caseKey':case['caseKey'], 'dataOrigin':'USER_CONFIRMED_AUTHORED_SYNTHETIC',
              'providerResponses':'REAL', 'maxBudgetCny':cap, 'status':'ERROR'}
    try:
        output = {'answer':case['answer'], 'claims':[{'text':text} for text in case['generatedClaims']],
                  'evidence':[{'sourceType':'DOCUMENT','text':case['evidenceText']}]}
        result = diagnose_answer(actor.tenant_id, job_id, output, case['question'], case['history'],
            lambda: EvaluationCatalog.check_snapshot(actor,snapshot['id'],check_current=True))
        labels=[]
        for index, assertion in enumerate(result.get('assertions', [])):
            # Only literal alignment to user-confirmed propositions; no LLM Gold.
            matches=[a for a in case['humanAssertions'] if a['text'].rstrip('。，,.；;') == assertion['text'].rstrip('。，,.；;')]
            if len(matches)==1:
                labels.append({'resultId':case['caseKey'],'assertionIndex':index,'support':matches[0]['support']})
        summary=calibration_summary([{'resultId':case['caseKey'],**result}],labels)
        record.update(status='COMPLETED',diagnosis=result,calibration=summary,
            limitations=['one authored counterexample, not production calibration or proven quality improvement',
                         'unaligned assertions have no human label; supplier invoice not checked'])
        assert result['missingFromGeneratedClaims'] and result['supported'] < result['denominator']
    finally:
        with get_db_connection() as conn:
            calls=conn.execute(select(model_call_table).where(model_call_table.c.job_id==job_id)).all()
        record.update(modelCalls=len(calls),localLedgerCny=sum(float(c.settled_amount) for c in calls if c.settled_amount is not None),
            unknownReservedCny=sum(float(c.reserved_amount) for c in calls if c.settled_amount is None))
        (root/'.local'/'real-assertion-acceptance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
