"""Opt-in synthetic Worker -> private Langfuse receipt and idempotency proof.

Run only against deploy/langfuse.compose.yml with fresh local-only credentials.
No external model/provider is called; the tracked provider result is synthetic.
"""
import json
import os
import time
import tracemalloc
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select, delete, update
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from test_evaluation import eval_env
from app import telemetry
from app.db import get_db_connection
from app.evaluation.catalog import EvaluationCatalog
from app.evaluation.runner import EvaluationRunner
from app.evaluation.langfuse_sync import sync_evaluation_scores
from app.evaluation.diagnostics import read_eval_diagnostics
from app.evaluation.metrics import EvaluationMetrics
from app.errors import ApiException
from app.identity.deps import Actor
from app.qa.model_usage import invoke_tracked
from app.qa.model_gateway import ModelGateway
from app.qa.service import QaService
from app.models import eval_run_table, eval_result_table, eval_review_table, eval_case_table, run_event_table, config_revision_table, kb_grant_table, app_user_table


@pytest.mark.skipif(os.getenv('RAG_LANGFUSE_PRIVATE_ACCEPTANCE') != '1', reason='requires isolated synthetic Langfuse')
def test_worker_trace_and_repeated_scores_received(monkeypatch, eval_env):
    import app.qa.service as qa_service
    import app.evaluation.runner as runner
    from app.runtime.worker import DurableJobWorker
    content = os.getenv('RAG_LANGFUSE_CONTENT_ACCEPTANCE') == '1'
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT' if content else 'METADATA_ONLY')
    for category in ('MODEL', 'RETRIEVAL', 'TOOL', 'QA'):
        monkeypatch.setenv('RAG_LANGFUSE_CAPTURE_' + category, 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    sdk = Langfuse(tracer_provider=provider, timeout=2, flush_interval=1,
                   should_export_span=telemetry._export_span, mask_otel_spans=telemetry._mask_otel_spans)
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    monkeypatch.setattr(telemetry, '_client', sdk)
    monkeypatch.setattr(telemetry, '_provider', provider)
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    monkeypatch.setattr(qa_service, 'is_test_env', lambda: False)
    actor = eval_env['admin']
    with get_db_connection() as conn:
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
            id='cfg-model-chat-main', kind='MODEL', name='synthetic pricing', revision=1,
            sha256=b'0'*32, status='PUBLISHED', payload={'modelAlias': 'chat-main', 'currency': 'CNY',
                'inputPricePerMillion': 1, 'outputPricePerMillion': 1}))
        conn.commit()
    calls = []

    def synthetic_chat(*, tenant_id, run_id=None, purpose, messages, model_alias='chat-main', **kwargs):
        assert purpose == 'ANSWER'
        evidence = json.loads(messages[-1]['content'])['evidence'][0]
        answer = json.dumps({'status': 'COMPLETED', 'claims': [{
            'id': 'c1', 'text': evidence['text'][:10],
            'evidence': [{'evidenceId': evidence['evidenceId'], 'start': 0, 'end': 10}],
        }], 'gaps': [], 'conflicts': []})
        def sent():
            calls.append('ANSWER')
            return SimpleNamespace(answer=answer)
        return invoke_tracked(sent, tenant_id=tenant_id, run_id=run_id, purpose=purpose,
            alias=model_alias, payload={'messages': messages, 'tools': []}, max_output_tokens=20,
            usage_reader=lambda _: {'input_tokens': 30, 'output_tokens': 10}).answer

    monkeypatch.setattr(ModelGateway, 'chat', synthetic_chat)
    def synthetic_chat_tools(*, tenant_id, run_id, purpose, messages, tools, **kwargs):
        tool_messages = [m for m in messages if m.get('role') == 'tool']
        if not tool_messages:
            reply = {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'synthetic-search', 'name': 'search_knowledge',
                'args': {'query': '知识库系统采用什么分块策略？', 'topK': 5}}]}
            if content:
                reply['tool_calls'].append({'id': 'synthetic-mock',
                    'name': 'get_final_price_rules', 'args': {}})
        else:
            evidence = json.loads(tool_messages[-1]['content'])['evidence'][0]
            reply = {'role': 'assistant', 'tool_calls': [], 'content': json.dumps({
                'status': 'COMPLETED', 'claims': [{'id': 'c1', 'text': evidence['excerpt'][:10],
                    'evidence': [{'evidenceId': evidence['evidenceId'], 'start': 0, 'end': 10}]}],
                'gaps': [], 'conflicts': []})}
        def sent():
            calls.append('AGENT')
            return reply
        return invoke_tracked(sent, tenant_id=tenant_id, run_id=run_id, purpose=purpose,
            alias='chat-main', payload={'messages': messages, 'tools': tools}, max_output_tokens=20,
            usage_reader=lambda _: {'input_tokens': 30, 'output_tokens': 10})
    monkeypatch.setattr(ModelGateway, 'chat_tools', synthetic_chat_tools)
    worker = DurableJobWorker(kinds={'QA_RUN'})
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: 0,
        sleep=lambda _: (QaService.recover_unfinished_runs(), worker.run_once())))
    snapshot = EvaluationCatalog.create_snapshot(actor, {'name': 'Synthetic receipt',
        'documentIds': [eval_env['doc_id']], 'approvedSyntheticOrRedacted': True})
    scope = {**snapshot['manifest'], 'snapshotId': snapshot['id']}
    suffix = uuid.uuid4().hex
    try:
        detail, ids = EvaluationRunner._run_qa_turns(actor, actor, 'eval-'+suffix, 'result-'+suffix,
            {'caseKey': 'synthetic', 'question': {'text': '知识库系统采用什么分块策略？'}},
            {'mode': 'RAG', 'configId': 'cfg-enterprise-retrieval-v2'}, scope)
        assert detail['status'] == 'COMPLETED'
        assert calls == ['AGENT', 'AGENT']
        qa_id = ids[-1]
        with get_db_connection() as conn:
            batch = conn.execute(select(run_event_table.c.payload).where(
                run_event_table.c.run_id == qa_id, run_event_table.c.stage == 'OBSERVABILITY_TRACE')).scalar()
        assert batch['sampled'] is True
        trace_id = batch['traceId']
        diagnostics = read_eval_diagnostics(actor, qa_id)
        actual = [d for d in diagnostics if d.get('stage') == 'MODEL_CONTEXT' and d.get('items')]
        assert actual and actual[0]['items'][0]['mappingStatus'] == 'EXACT'

        if os.getenv('RAG_LANGFUSE_FAULT_ACCEPTANCE') == '1':
            started = time.perf_counter()
            telemetry.shutdown_telemetry(timeout_seconds=0.2)
            elapsed = time.perf_counter() - started
            assert elapsed < 0.5
            target = Path(__file__).resolve().parents[2] / '.local' / 'langfuse-outage.json'
            target.write_text(json.dumps({'qaStatus': detail['status'], 'modelCalls': len(calls),
                'shutdownSeconds': round(elapsed, 3), 'telemetryStatus': telemetry.telemetry_status(),
                'fault': 'web and worker stopped', 'provider': 'synthetic tracked fixture'}, indent=2), encoding='utf-8')
            return

        dataset = EvaluationCatalog.create_dataset(actor, {'name': 'Synthetic score receipt'})
        EvaluationCatalog.save_cases(actor, dataset['id'], [{'caseKey': 'receipt', 'split': 'TEST',
            'groupKey': 'receipt', 'category': 'FACT', 'question': {'text': 'synthetic'},
            'persona': {'loginName': 'admin'}, 'expected': {'status': 'ANSWER', 'evidenceGroups': []},
            'annotationStatus': 'AUTO_REVIEWED'}])
        with get_db_connection() as conn:
            case_id = conn.execute(select(eval_case_table.c.id).where(
                eval_case_table.c.dataset_version_id == dataset['id'])).scalar()
            eval_id, result_id = str(uuid.uuid4()), str(uuid.uuid4())
            conn.execute(eval_run_table.insert().values(tenant_id=actor.tenant_id, id=eval_id,
                dataset_version_id=dataset['id'], snapshot_id=snapshot['id'], config_id='cfg-enterprise-retrieval-v2',
                job_id=str(uuid.uuid4()), status='COMPLETED', manifest_json={'rubricVersion': 'v2'}))
            conn.execute(eval_result_table.insert().values(tenant_id=actor.tenant_id, id=result_id,
                eval_run_id=eval_id, dataset_version_id=dataset['id'], case_id=case_id, qa_run_id=qa_id,
                status='COMPLETED', metrics_json={'hit': 1.0, 'recall': None},
                judge_json={'correctness': 'UNDETERMINED'}))
            for label in ('CORRECT', 'WRONG'):
                conn.execute(eval_review_table.insert().values(tenant_id=actor.tenant_id, id=str(uuid.uuid4()),
                    result_id=result_id, reviewer_id=actor.user_id, labels_json={'correctness': label},
                    reason_text='SYNTHETIC_PRIVATE_REVIEW_CANARY'))
            conn.commit()
        first = sync_evaluation_scores(actor, eval_id)
        second = sync_evaluation_scores(actor, eval_id)
        assert first['sent'] == second['sent'] == 4
        sdk.flush()
        observations, scores = [], []
        start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            observations = sdk.api.observations.get_many(trace_id=trace_id,
                from_start_time=start_time, to_start_time=datetime.now(timezone.utc),
                fields='core,basic,io,metadata,model,usage', limit=100).data
            scores = sdk.api.scores_v3.get_many_v3(trace_id=trace_id,
                from_timestamp=start_time, to_timestamp=datetime.now(timezone.utc)).data
            if len(scores) == 4 and len([o for o in observations if o.type == 'GENERATION']) == 2:
                break
            time.sleep(1)
        assert len(scores) == 4
        assert len({s.id for s in scores}) == 4
        generations = [o for o in observations if o.type == 'GENERATION']
        assert len(generations) == 2
        if content:
            required = {'qa.worker.attempt', 'agent.run', 'agent.model', 'agent.tool',
                        'rag.retrieve', 'rag.prepare_query', 'rag.bm25_recall',
                        'rag.context_budget', 'rag.finalize_hits', 'model.agent'}
            assert required.issubset({o.name for o in observations})
            assert all(o.input is not None and o.output is not None for o in observations if o.name in required)
            # This pinned server's observations API returns serialized JSON bodies.
            def body(value):
                return json.loads(value) if isinstance(value, str) else value
            root = next(o for o in observations if o.name == 'qa.worker.attempt')
            assert body(root.input)['question'] == detail['question'] and body(root.output)['answer'] == detail['answer']
            retrieved = next(o for o in observations if o.name == 'rag.finalize_hits')
            assert eval_env['quote'] in str(retrieved.output)
            assert body(generations[0].input)['messages'] and body(generations[0].input)['tools']
            assert any(o.name == 'agent.tool' and body(o.output).get('mock') is True
                       for o in observations if o.output is not None)
        else:
            assert all(o.input is None and o.output is None for o in observations)
        assert 'SYNTHETIC_PRIVATE_REVIEW_CANARY' not in str([o.model_dump() for o in observations]) + str([s.model_dump() for s in scores])
        assert all(g.usage_details['input'] == 30 and g.usage_details['output'] == 10 for g in generations)
        # Provider SDK failure cannot modify authoritative evaluation data.
        monkeypatch.setattr(sdk, 'create_score', lambda **_: (_ for _ in ()).throw(RuntimeError('PRIVATE_CANARY')))
        failed = sync_evaluation_scores(actor, eval_id)
        assert failed['failed'] == 4 and failed['sent'] == 0
        with get_db_connection() as conn:
            assert conn.execute(select(eval_result_table.c.metrics_json).where(
                eval_result_table.c.id == result_id)).scalar() == {'hit': 1.0, 'recall': None}

        # Paired serial comparison includes local diagnostics in both modes.
        measurements = {'off': [], 'on': []}
        peaks = {'off': [], 'on': []}
        for index in range(13):  # first pair warms the Worker/graph caches
            for mode in (('off', 'on') if index % 2 == 0 else ('on', 'off')):
                monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true' if mode == 'on' else 'false')
                before = len(calls)
                tracemalloc.start()
                started = time.perf_counter()
                compared, _ = EvaluationRunner._run_qa_turns(actor, actor,
                    'perf-eval-'+suffix+mode+str(index), 'perf-result-'+suffix+mode+str(index),
                    {'caseKey': 'synthetic', 'question': {'text': '知识库系统采用什么分块策略？'}},
                    {'mode': 'RAG', 'configId': 'cfg-enterprise-retrieval-v2'}, scope)
                duration = (time.perf_counter() - started) * 1000
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert compared['status'] == detail['status'] and compared['answer'] == detail['answer']
                assert len(calls) - before == 2
                if index:
                    measurements[mode].append(duration)
                    peaks[mode].append(peak)
        monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true')
        # Waiting for approval ends the physical lease; resuming creates a new trace.
        monkeypatch.setenv('AGENT_APPROVAL_TOOLS', 'search_knowledge')
        conversation = QaService.create_conversation(actor, {'title': 'Synthetic approval'})
        approval_run = QaService.submit_run(actor, conversation['conversationId'],
            {'question': '知识库系统采用什么分块策略？', 'segmentId': conversation['currentSegmentId']},
            idempotency_key='approval-'+suffix, trusted_config_id='cfg-enterprise-retrieval-v2',
            evaluation_scope=scope)['runId']
        QaService.recover_unfinished_runs()
        assert worker.run_once()
        assert QaService.get_run(actor, approval_run)['status'] == 'WAITING_APPROVAL'
        QaService.resolve_tool_approval(actor, approval_run,
            {'toolCallId': 'synthetic-search', 'decision': 'APPROVE'})
        QaService.recover_unfinished_runs()
        assert worker.run_once()
        assert QaService.get_run(actor, approval_run)['status'] == 'COMPLETED'
        with get_db_connection() as conn:
            approval_batches = conn.execute(select(run_event_table.c.payload).where(
                run_event_table.c.run_id == approval_run, run_event_table.c.stage == 'OBSERVABILITY_TRACE')
                .order_by(run_event_table.c.sequence_no)).scalars().all()
        assert len(approval_batches) == 2
        assert approval_batches[0]['traceId'] != approval_batches[1]['traceId']
        assert approval_batches[0]['leaseEpoch'] < approval_batches[1]['leaseEpoch']
        monkeypatch.delenv('AGENT_APPROVAL_TOOLS')
        # Disabled persona skips export even when the caller retains admin rights.
        with get_db_connection() as conn:
            conn.execute(update(app_user_table).where(app_user_table.c.id == actor.user_id).values(status='DISABLED'))
            conn.execute(update(app_user_table).where(app_user_table.c.id == 'eval_operator_id').values(is_admin=True))
            conn.commit()
        maintainer = Actor(actor.tenant_id, 'eval_operator_id', True)
        revoked = sync_evaluation_scores(maintainer, eval_id)
        assert revoked['skipped'] == 1 and revoked['sent'] == revoked['failed'] == 0
        with pytest.raises(ApiException):
            read_eval_diagnostics(maintainer, qa_id)
        with get_db_connection() as conn:
            conn.execute(update(app_user_table).where(app_user_table.c.id == actor.user_id).values(status='ACTIVE'))
            conn.execute(delete(kb_grant_table).where(kb_grant_table.c.tenant_id == actor.tenant_id))
            conn.commit()
        with pytest.raises(ApiException) as revoked_snapshot:
            sync_evaluation_scores(actor, eval_id)
        assert revoked_snapshot.value.status_code == 403
        with pytest.raises(ApiException):
            read_eval_diagnostics(actor, qa_id)
        evidence = {'traceId': trace_id, 'modelCalls': 2, 'comparisonModelCalls': len(calls)-2, 'receivedGenerations': len(generations),
                    'receivedScores': len(scores), 'duplicateSyncEnqueues': second['sent'],
                    'stages': sorted({d.get('stage') for d in diagnostics if d.get('stage')}),
                    'actualContextMapping': actual[0]['items'][0]['mappingStatus'],
                    'dataMode': 'CONTENT' if content else 'METADATA_ONLY', 'provider': 'synthetic tracked fixture',
                    'contentCategoriesVerified': ['MODEL', 'RETRIEVAL', 'TOOL', 'QA'] if content else [],
                    'mockToolBodyVerified': content,
                    'scoreFailurePreservedMetrics': True, 'revokedPersonaSkipped': True,
                    'hitlDistinctBatches': len(approval_batches),
                    'performance': {mode: {'samples': len(values), 'concurrency': 1,
                        'p50Ms': round(EvaluationMetrics.percentile(values, .5), 2),
                        'p95Ms': round(EvaluationMetrics.percentile(values, .95), 2),
                        'maxPythonAllocatedBytes': max(peaks[mode])} for mode, values in measurements.items()}}
        target = Path(__file__).resolve().parents[2] / '.local' / (
            'langfuse-content-receipt.json' if content else 'langfuse-receipt.json')
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    finally:
        sdk.shutdown()
