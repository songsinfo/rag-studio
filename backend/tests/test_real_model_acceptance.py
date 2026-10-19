"""Opt-in paid synthetic Worker/provider/Langfuse acceptance, capped by caller."""
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy import select

from app import telemetry
from app.db import get_db_connection
from app.evaluation.catalog import EvaluationCatalog
from app.evaluation.judge import judge_answer, RUBRIC_VERSION
from app.evaluation.runner import EvaluationRunner
from app.models import config_revision_table, job_table, eval_run_table, model_call_table, run_event_table
from app.runtime.worker import DurableJobWorker
from test_evaluation import eval_env


@pytest.mark.skipif(os.getenv('RAG_REAL_MODEL_ACCEPTANCE') != '1', reason='explicit paid provider acceptance only')
def test_real_worker_model_usage_and_private_receipt(monkeypatch, eval_env):
    import app.qa.service as service
    import app.evaluation.runner as runner
    root = Path(__file__).resolve().parents[2]
    gateway = dotenv_values(root / '.env')
    private = dotenv_values(root / '.local/langfuse-eval.env')
    cap = float(os.environ['RAG_REAL_MODEL_MAX_COST_CNY'])
    assert 0 < cap <= 5
    assert gateway.get('LITELLM_BASE_URL') and gateway.get('LITELLM_MASTER_KEY')
    monkeypatch.setenv('LITELLM_BASE_URL', gateway['LITELLM_BASE_URL'])
    monkeypatch.setenv('LITELLM_MASTER_KEY', gateway['LITELLM_MASTER_KEY'])
    monkeypatch.setenv('RAG_MONTHLY_BUDGET_CNY', str(cap))
    monkeypatch.setenv('AGENT_CALL_MAX_ATTEMPTS', '1')
    monkeypatch.setenv('AGENT_MAX_MODEL_CALLS', '3')
    monkeypatch.setenv('AGENT_MAX_ROUNDS', '3')
    monkeypatch.setenv('AGENT_FALLBACK_MODEL_ALIAS', '')
    for key, value in private.items():
        if key.startswith(('LANGFUSE_', 'RAG_LANGFUSE_')):
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(service, 'is_test_env', lambda: False)
    provider = TracerProvider(shutdown_on_exit=False)
    sdk = Langfuse(tracer_provider=provider, timeout=2, flush_interval=1,
        should_export_span=telemetry._export_span, mask_otel_spans=telemetry._mask_otel_spans)
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    monkeypatch.setattr(telemetry, '_client', sdk)
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    actor = eval_env['admin']
    suffix = uuid.uuid4().hex
    job_id, eval_id = str(uuid.uuid4()), str(uuid.uuid4())
    with get_db_connection() as conn:
        # Conservative local acceptance rates, not supplier tariff or invoice.
        for alias in ('chat-main', 'judge-main'):
            conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
                id='cfg-model-'+alias, kind='MODEL', name='acceptance budget ceiling', revision=1,
                sha256=b'0'*32, status='PUBLISHED', payload={'modelAlias': alias, 'currency': 'CNY',
                    'inputPricePerMillion': 100, 'outputPricePerMillion': 200}))
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
            id='real-acceptance-bm25', kind='RETRIEVAL', name='synthetic BM25', revision=1,
            sha256=b'0'*32, status='PUBLISHED', payload={'retrieval': {'mode': 'BM25'},
                'query': {'rewrite': False, 'hyde': False, 'multiQuery': False},
                'post': {'rerank': False, 'mmr': False}}))
        conn.execute(job_table.insert().values(tenant_id=actor.tenant_id, id=job_id,
            kind='EVALUATE', target_id=eval_id, requested_by=actor.user_id,
            idempotency_key='real-'+suffix, request_hash=b'0'*32, status='RUNNING',
            checkpoint={'maxCostCny': cap, 'maxModelCalls': 4}))
        conn.commit()
    snapshot = EvaluationCatalog.create_snapshot(actor, {'name': 'Synthetic real-provider acceptance',
        'documentIds': [eval_env['doc_id']], 'approvedSyntheticOrRedacted': True})
    scope = {**snapshot['manifest'], 'snapshotId': snapshot['id']}
    with get_db_connection() as conn:
        conn.execute(eval_run_table.insert().values(tenant_id=actor.tenant_id, id=eval_id,
            dataset_version_id='synthetic-acceptance', snapshot_id=snapshot['id'], config_id='real-acceptance-bm25',
            job_id=job_id, status='RUNNING', manifest_json={'rubricVersion': 'v2'}))
        conn.commit()
    worker = DurableJobWorker(kinds={'QA_RUN'}, max_attempts=1)
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: 0,
        sleep=lambda _: worker.run_once()))
    try:
        detail, run_ids = EvaluationRunner._run_qa_turns(actor, actor, eval_id, 'real-result-'+suffix,
            {'caseKey': 'real-synthetic', 'question': {'text': '请先检索“结构化分块策略”，回答资料原文开头的前五个汉字是什么。只引用原文前五字，即区间 [0,5)，不要引用包装标记。'}},
            {'mode': 'AGENT', 'configId': 'real-acceptance-bm25'}, scope)
        assert detail['status'] == 'COMPLETED', detail.get('terminalReason')
        qa_id = run_ids[-1]
        with telemetry.observation_batch({'kind': 'EVALUATE', 'tenant_id': actor.tenant_id,
                'id': job_id, 'target_id': eval_id, 'lease_epoch': 1}) as judge_batch:
            judgment = judge_answer(actor.tenant_id, qa_id, detail['question'],
                {'status': 'ANSWER', 'factPoints': ['知识库系统'], 'evidenceGroups': []},
                detail, conversation_context=EvaluationRunner._judge_conversation_context(actor, run_ids))
        assert judgment['rubricVersion'] == RUBRIC_VERSION
        with get_db_connection() as conn:
            calls = conn.execute(select(model_call_table).where(model_call_table.c.run_id == qa_id)).all()
            batch = conn.execute(select(run_event_table.c.payload).where(run_event_table.c.run_id == qa_id,
                run_event_table.c.stage == 'OBSERVABILITY_TRACE')).scalar()
        assert 2 <= len(calls) <= 4 and all(c.status == 'SETTLED' for c in calls)
        assert all(c.usage_detail['usageEstimator'] == 'provider' for c in calls)
        amount = sum(float(c.settled_amount) for c in calls)
        assert amount <= cap
        sdk.flush()
        deadline = time.monotonic()+45
        while True:
            observations = sdk.api.observations.get_many(trace_id=batch['traceId'],
                from_start_time=datetime.now(timezone.utc)-timedelta(minutes=10),
                to_start_time=datetime.now(timezone.utc), fields='core,basic,io,usage', limit=100).data
            generations = [o for o in observations if o.type == 'GENERATION']
            agent_calls = [c for c in calls if c.purpose == 'AGENT']
            judge_observations = sdk.api.observations.get_many(trace_id=judge_batch.trace_id,
                from_start_time=datetime.now(timezone.utc)-timedelta(minutes=10),
                to_start_time=datetime.now(timezone.utc), fields='core,basic,io,usage', limit=100).data
            judge_generations = [o for o in judge_observations if o.type == 'GENERATION']
            if len(generations) == len(agent_calls) and len(judge_generations) == 1:
                break
            assert time.monotonic() < deadline
            time.sleep(1)
        assert generations and all(g.input is None and g.output is None for g in generations+judge_generations)
        record = {'dataOrigin': 'SYNTHETIC', 'providerResponses': 'REAL', 'qaStatus': detail['status'],
            'modelCalls': len(calls), 'receivedAgentGenerations': len(generations), 'traceId': batch['traceId'],
            'receivedJudgeGenerations': len(judge_generations), 'judgeTraceId': judge_batch.trace_id,
            'judgeParsed': True, 'judgeRubricVersion': judgment['rubricVersion'],
            'maxBudgetCny': cap, 'localLedgerCny': round(amount, 6),
            'pricingBasis': 'conservative acceptance ceiling 100/200 CNY per million, not supplier invoice',
            'inputTokens': sum(c.input_tokens for c in calls), 'outputTokens': sum(c.output_tokens for c in calls),
            'cacheReadTokens': [c.usage_detail.get('cacheReadTokens') for c in calls],
            'limitations': ['SQLite/BM25 synthetic corpus; no production MySQL/Milvus load',
                            'No human Gold calibration; no judge quality claim', 'Supplier invoice not checked']}
        target = root / '.local' / 'real-model-acceptance.json'
        target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    finally:
        with get_db_connection() as conn:
            audit = conn.execute(select(model_call_table.c.settled_amount, model_call_table.c.reserved_amount, model_call_table.c.status)
                .where(model_call_table.c.tenant_id == actor.tenant_id)).all()
        accounting = {'calls': len(audit), 'localLedgerCny': sum(float(c.settled_amount) for c in audit if c.settled_amount is not None),
                      'unknownCalls': sum(c.settled_amount is None for c in audit),
                      'unknownReservedCny': sum(float(c.reserved_amount) for c in audit if c.settled_amount is None),
                      'maxBudgetCny': cap}
        (root / '.local' / 'real-model-latest-accounting.json').write_text(json.dumps(accounting, indent=2), encoding='utf-8')
        sdk.shutdown()
        provider.shutdown()
