"""Synthetic privacy, fault and concurrency checks at the exporter boundary."""
import gzip
import json
import threading
import time
from queue import Full
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetry
from app.evaluation.diagnostics import model_context_snapshot
from test_evaluation import eval_env


def test_private_diagnostic_io_cannot_send_after_run_is_cancelled(monkeypatch, eval_env):
    from sqlalchemy import select
    from app.db import get_db_connection
    from app.errors import ApiException
    from app.models import config_revision_table, model_call_table
    from app.qa.model_usage import invoke_tracked
    from app.qa.service import QaService
    import app.qa.service as service
    import app.evaluation.diagnostics as diagnostics
    actor = eval_env['admin']
    monkeypatch.setattr(service, 'is_test_env', lambda: False)
    with get_db_connection() as conn:
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
            id='cfg-model-chat-main', kind='MODEL', name='synthetic pricing', revision=1,
            sha256=b'0'*32, status='PUBLISHED', payload={'modelAlias': 'chat-main', 'currency': 'CNY',
                'inputPricePerMillion': 1, 'outputPricePerMillion': 1}))
        conn.commit()
    conversation = QaService.create_conversation(actor, {'title': 'synthetic cancellation'})
    run_id = QaService.submit_run(actor, conversation['conversationId'],
        {'question': '合成问题', 'segmentId': conversation['currentSegmentId']}, idempotency_key='cancel-during-io')['runId']
    def delayed_capture(*args):
        QaService.cancel_run(actor, run_id)
        return True
    monkeypatch.setattr(diagnostics, 'capture_model_context', delayed_capture)
    sent = []
    with pytest.raises(ApiException) as cancelled:
        invoke_tracked(lambda: sent.append(True), tenant_id=actor.tenant_id, run_id=run_id,
            purpose='ANSWER', alias='chat-main', payload={'messages': []})
    assert cancelled.value.code == 'CANCELLED' and not sent
    with get_db_connection() as conn:
        call = conn.execute(select(model_call_table).where(model_call_table.c.run_id == run_id)).one()
    assert call.status == 'NOT_SENT' and call.input_tokens == call.output_tokens == 0


def test_tracked_retry_and_fallback_have_one_generation_per_actual_call(monkeypatch, eval_env):
    import httpx
    from sqlalchemy import select
    from opentelemetry import trace
    from app.db import get_db_connection
    from app.models import config_revision_table, model_call_table, job_table
    from app.qa.model_gateway import ModelGateway
    import app.qa.model_gateway as gateway
    actor = eval_env['admin']
    with get_db_connection() as conn:
        conn.execute(job_table.insert().values(tenant_id=actor.tenant_id, id='synthetic-retry-job',
            kind='EVALUATE', target_id='synthetic', requested_by=actor.user_id,
            idempotency_key='retry-check', request_hash=b'0'*32, status='RUNNING'))
        for alias in ('chat-main', 'synthetic-fallback'):
            conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
                id='cfg-model-'+alias, kind='MODEL', name='synthetic pricing', revision=1,
                sha256=b'0'*32, status='PUBLISHED', payload={'modelAlias': alias, 'currency': 'CNY',
                    'inputPricePerMillion': 1, 'outputPricePerMillion': 1}))
        conn.commit()
    sent = []
    def model(alias, *args):
        def invoke(messages):
            sent.append(alias)
            if alias == 'chat-main':
                raise httpx.ReadTimeout('PRIVATE_CANARY')
            return SimpleNamespace(content='synthetic answer', usage_metadata={'input_tokens': 4, 'output_tokens': 0})
        result = SimpleNamespace(invoke=invoke)
        result.bind = lambda **kwargs: result
        return result
    monkeypatch.setattr(gateway, 'get_chat_model', model)
    monkeypatch.setenv('LITELLM_BASE_URL', 'http://model.test')
    monkeypatch.setenv('AGENT_CALL_MAX_ATTEMPTS', '2')
    monkeypatch.setenv('AGENT_FALLBACK_MODEL_ALIAS', 'synthetic-fallback')
    monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('RAG_LANGFUSE_SAMPLE_RATE', '1')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    sdk = Langfuse(public_key='pk-lf-retry-check', secret_key='sk-lf-retry-check',
        base_url='http://127.0.0.1:1', tracer_provider=provider, span_exporter=exporter,
        should_export_span=telemetry._export_span, mask_otel_spans=telemetry._mask_otel_spans)
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    monkeypatch.setattr(telemetry, '_client', sdk)
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    try:
        with telemetry.observation_batch({'kind': 'EVALUATE', 'tenant_id': actor.tenant_id, 'target_id': 'synthetic'}):
            assert ModelGateway.chat(actor.tenant_id, None, 'ANSWER',
                [{'role': 'user', 'content': 'SYNTHETIC_PRIVATE_CANARY'}], job_id='synthetic-retry-job') == 'synthetic answer'
        sdk.flush()
        generations = [s for s in exporter.get_finished_spans() if s.name == 'model.answer']
        assert sent == ['chat-main', 'chat-main', 'synthetic-fallback']
        assert len(generations) == len(sent)
        with get_db_connection() as conn:
            rows = conn.execute(select(model_call_table).where(model_call_table.c.tenant_id == actor.tenant_id)).all()
        assert len(rows) == 3 and len({r.id for r in rows}) == 3
        assert [s.attributes['langfuse.observation.model.name'] for s in generations] == sent
        assert 'SYNTHETIC_PRIVATE_CANARY' not in str([dict(s.attributes) for s in generations])
        assert generations[-1].attributes['langfuse.observation.metadata.cacheStatus'] == 'UNKNOWN'
        assert generations[-1].attributes['langfuse.observation.metadata.costStatus'] == 'SETTLED'
    finally:
        sdk.shutdown()


@pytest.mark.parametrize('cause', ['rerank', 'context_budget'])
def test_pipeline_snapshots_distinguish_lost_evidence_without_changing_ranking(monkeypatch, eval_env, cause):
    from app.retrieval.pipeline import RetrievalPipeline
    from app.retrieval.config import RetrievalConfig
    import app.retrieval.pipeline as pipeline
    from app.evaluation.diagnostics import candidate_snapshot
    actor = eval_env['admin']
    cfg = RetrievalConfig('synthetic', mode='HYBRID', rerank_enabled=cause == 'rerank',
                          context_max_bytes=1 if cause == 'context_budget' else 6000)
    monkeypatch.setattr(pipeline, 'load_retrieval_config', lambda *a: cfg)
    monkeypatch.setattr(pipeline, 'rerank_hits', lambda **kwargs: ([], []))
    stages = []
    def capture(stage, hits, limit, route_index):
        stages.append(candidate_snapshot(stage, hits, limit=limit, retrieval_index=1, route_index=route_index))
    baseline = RetrievalPipeline.retrieve(actor, '结构化分块策略')
    observed = RetrievalPipeline.retrieve(actor, '结构化分块策略', diagnostic_fn=capture)
    assert baseline == observed
    by_stage = {s['stage']: s for s in stages}
    assert by_stage['dense_recall']['items'] and by_stage['bm25_recall']['items']
    assert by_stage['rrf_fusion']['items']
    assert bool(by_stage['rerank']['items']) == (cause != 'rerank')
    assert not by_stage['context_budget']['items']
    assert eval_env['quote'] not in str(stages)
    def fail(*args):
        raise RuntimeError('PRIVATE_CANARY')
    assert RetrievalPipeline.retrieve(actor, '结构化分块策略', diagnostic_fn=fail) == baseline


def test_actual_context_records_only_sent_excerpt_and_exact_ranges():
    full = 'prefix-FIRST-EVIDENCE-second-TAIL'
    source = {'evidenceId': 'E1', 'documentId': 'doc', 'buildId': 'build',
              'span': {'chunkId': 'chunk', 'sourceUnits': [
                  {'unitId': 'u', 'contextStart': 0, 'contextEnd': len(full), 'contextMapped': True}]}}
    payload = {'messages': [{'role': 'tool', 'content': json.dumps({'evidence': [
        {'evidenceId': 'E1', 'documentId': 'doc', 'buildId': 'build',
         'excerpt': 'FIRST-EVIDENCE', 'truncated': True}]})}]}
    snap = model_context_snapshot(payload, [source], {('build', 'chunk'): full})
    assert snap['contextState'] == 'ACTUAL_SEND'
    assert snap['items'][0]['contextStart'] == 7
    assert snap['items'][0]['contextEnd'] == 21
    assert snap['items'][0]['sourceUnits'][0]['contextStart'] == 7
    assert snap['items'][0]['truncated'] is True
    assert 'FIRST-EVIDENCE' not in str(snap)
    assert snap['complete'] is False  # history is not reconstructed from citations
    payload['messages'][0]['content'] = payload['messages'][0]['content'].replace('FIRST-EVIDENCE', 'transformed')
    assert model_context_snapshot(payload, [source], {('build', 'chunk'): full})['items'][0]['mappingStatus'] == 'UNKNOWN'


def test_guardrail_wrapped_prefix_and_ambiguous_context_ranges():
    full = 'unique source fragment after prefix'
    source = {'evidenceId': 'E1', 'documentId': 'doc', 'buildId': 'build',
              'span': {'chunkId': 'child', 'parentChunkId': 'parent', 'guardrail': {}}}
    payload = {'messages': [{'role': 'tool', 'content': json.dumps({'evidence': [{
        'evidenceId': 'E1', 'documentId': 'doc', 'buildId': 'build', 'truncated': True,
        'excerpt': '<retrieved_document trust="untrusted-data">\nunique source'}]})}]}
    item = model_context_snapshot(payload, [source], {('build', 'parent'): full})['items'][0]
    assert item['mappingStatus'] == 'EXACT' and item['contextChunkId'] == 'parent'
    assert item['contextEnd'] == len('unique source')
    item = model_context_snapshot(payload, [source], {('build', 'parent'): full + full})['items'][0]
    assert item['mappingStatus'] == 'UNKNOWN'


def test_span_faults_do_not_repeat_or_swallow_business_exception(monkeypatch):
    from app.observability import agent_span
    from opentelemetry import trace
    calls = []
    monkeypatch.setattr(trace, 'get_tracer', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('PRIVATE_CANARY')))
    with pytest.raises(ValueError, match='business'):
        with agent_span('agent.model'):
            calls.append('sent')
            raise ValueError('business')
    assert calls == ['sent']


def test_enabled_batches_are_fresh_tenant_scoped_and_sampled_locally(monkeypatch):
    from opentelemetry import trace
    provider = TracerProvider(shutdown_on_exit=False)
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    monkeypatch.setattr(telemetry, '_client', object())
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('RAG_LANGFUSE_SAMPLE_RATE', '1')
    monkeypatch.setenv('RAG_LANGFUSE_ID_HMAC_KEY', 'x' * 40)
    job = {'kind': 'QA_RUN', 'id': 'same-job', 'target_id': 'same-run', 'tenant_id': 'tenant-a',
           'attempt': 0, 'lease_epoch': 1}
    try:
        with provider.get_tracer('external').start_as_current_span('ambient'):
            with telemetry.observation_batch(job) as first:
                assert first.sampled and first.job_attempt == 1
                assert trace.get_current_span().parent is None
                a = telemetry.identifier('run', 'same-run')
            with telemetry.observation_batch({**job, 'attempt': 1, 'lease_epoch': 2,
                                              'tenant_id': 'tenant-b'}) as second:
                assert second.trace_id != first.trace_id
                assert second.job_attempt == 2 and second.lease_epoch == 2
                assert telemetry.identifier('run', 'same-run') != a
        monkeypatch.setenv('RAG_LANGFUSE_SAMPLE_RATE', '0')
        with telemetry.observation_batch(job) as unsampled:
            assert not unsampled.sampled and not telemetry.export_current_span()
        assert telemetry.active_batch() is None
    finally:
        provider.shutdown()


def test_process_initialization_is_idempotent_and_missing_config_stays_local(monkeypatch):
    from opentelemetry import trace
    import langfuse
    provider = TracerProvider(shutdown_on_exit=False)
    monkeypatch.setattr(trace, 'get_tracer_provider', lambda: provider)
    for name, value in (('_configured', False), ('_client', None), ('_provider', None), ('_status', 'DISABLED')):
        monkeypatch.setattr(telemetry, name, value)
    for name in ('OTEL_EXPORTER_OTLP_ENDPOINT', 'RAG_LANGFUSE_ENABLED', 'LANGFUSE_BASE_URL',
                 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'):
        monkeypatch.delenv(name, raising=False)
    telemetry.configure_telemetry()
    assert telemetry.telemetry_status() == 'DISABLED'
    monkeypatch.setattr(telemetry, '_configured', False)
    monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true')
    telemetry.configure_telemetry()
    assert telemetry.telemetry_status() == 'MISSING_CONFIG'
    assert not provider._active_span_processor._span_processors
    monkeypatch.setattr(telemetry, '_configured', False)
    monkeypatch.setenv('LANGFUSE_BASE_URL', 'http://127.0.0.1:1')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-init-check')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-init-check')
    monkeypatch.setattr(langfuse, 'Langfuse', lambda **kwargs: Langfuse(
        **kwargs, span_exporter=InMemorySpanExporter()))
    try:
        for _ in range(3):
            telemetry.configure_telemetry()
        assert telemetry.telemetry_status() == 'ENABLED'
        assert len(provider._active_span_processor._span_processors) == 1
    finally:
        telemetry.shutdown_telemetry()


@pytest.mark.parametrize('failure', [Full, TimeoutError, ValueError])
def test_sdk_setup_failures_do_not_repeat_business(monkeypatch, failure):
    def fail(**kwargs):
        raise failure('PRIVATE_CANARY')
    monkeypatch.setattr(telemetry, 'export_current_span', lambda: True)
    monkeypatch.setattr(telemetry, '_client', SimpleNamespace(start_observation=fail))
    calls = []
    with telemetry.model_observation('AGENT', 'chat-main', 'call') as observation:
        assert observation is None
        calls.append('sent')
    assert calls == ['sent']


def test_shutdown_wait_is_bounded_and_sdk_shutdown_is_called(monkeypatch):
    release, called, finished = threading.Event(), threading.Event(), threading.Event()
    def close():
        called.set()
        release.wait(2)
        finished.set()
    monkeypatch.setattr(telemetry, '_client', SimpleNamespace(shutdown=close))
    monkeypatch.setattr(telemetry, '_provider', None)
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    started = time.perf_counter()
    telemetry.shutdown_telemetry(.05)
    assert time.perf_counter() - started < .5
    assert called.is_set() and telemetry.telemetry_status() == 'SHUTDOWN_TIMEOUT'
    release.set()
    assert finished.wait(1)


def test_parallel_graph_nodes_have_one_parent_per_retrieval(monkeypatch):
    from app.retrieval.rag_graph import RagRetrievalGraph
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    try:
        def retrieval(index):
            with telemetry.safe_span('rag.retrieve') as root:
                runtime = SimpleNamespace(context=SimpleNamespace(
                    retrieval_index=index, observation_context=copy_context()))
                node = RagRetrievalGraph._observe_stage('prepare_query', lambda *a: {})
                with ThreadPoolExecutor(max_workers=2) as nodes:
                    list(nodes.map(lambda _: node({}, runtime), range(2)))
                return root.get_span_context().trace_id, root.get_span_context().span_id
        with ThreadPoolExecutor(max_workers=8) as pool:
            parents = dict(pool.map(retrieval, range(24)))
        assert len(parents) == 24
        children = [s for s in exporter.get_finished_spans() if s.name == 'rag.prepare_query']
        assert len(children) == 48
        assert all(s.parent.span_id == parents[s.context.trace_id] for s in children)
    finally:
        provider.shutdown()


def test_real_otlp_http_payload_contains_no_content_or_raw_ids():
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from google.protobuf.json_format import MessageToDict
    received = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            if self.headers.get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            parsed = ExportTraceServiceRequest()
            parsed.ParseFromString(body)
            received.append(MessageToDict(parsed))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = TracerProvider()
    sdk = Langfuse(public_key='pk-lf-wire-check', secret_key='sk-lf-wire-check',
                   base_url=f'http://127.0.0.1:{server.server_port}', tracer_provider=provider,
                   should_export_span=telemetry._export_span, mask_otel_spans=telemetry._mask_otel_spans,
                   timeout=1, flush_interval=600)
    try:
        with provider.get_tracer('rag.agent').start_as_current_span('qa.worker.attempt') as root:
            root.set_attribute('rag.export', True)
            obs = sdk.start_observation(as_type='generation', name='model.answer', model='chat-main',
                                        metadata={'purpose': 'ANSWER', 'call': 'b'*32})
            telemetry.finish_model_observation(obs, status='SUCCESS', duration_ms=1.2,
                reported_usage={'input': 4, 'output': 0}, usage_source='PROVIDER',
                settlement={'settledAmount': '0.000001'})
            unsafe = sdk.start_observation(as_type='generation', name='model.unsafe',
                input='SYNTHETIC_PRIVATE_PROMPT', output='SYNTHETIC_PRIVATE_ANSWER',
                metadata={'headers': {'Authorization': 'SYNTHETIC_PRIVATE_KEY'},
                          'documentId': 'SYNTHETIC_RAW_DOCUMENT_ID', 'title': 'SYNTHETIC_PRIVATE_TITLE'})
            unsafe.end()
            with provider.get_tracer('rag.agent').start_as_current_span('agent.unsafe') as bad:
                bad.set_attribute('rag.export', True)
                bad.set_attribute('rag.run_hmac', 'SYNTHETIC_RAW_RUN_ID')
                bad.set_attribute('path', 'SYNTHETIC_PRIVATE_PATH')
                bad.add_event('exception', {'exception.message': 'SYNTHETIC_PRIVATE_CANARY'})
            with provider.get_tracer('sqlalchemy').start_as_current_span('sql') as bad:
                bad.set_attribute('db.statement', 'SYNTHETIC_PRIVATE_CANARY')
        sdk.flush()
        serialized = json.dumps(received)
        assert received
        assert 'model.answer' in serialized
        assert 'settledCny' in serialized
        assert 'SYNTHETIC_PRIVATE_CANARY' not in serialized
        assert 'agent.unsafe' not in serialized
        assert 'model.unsafe' not in serialized
        assert not any(canary in serialized for canary in ('SYNTHETIC_PRIVATE_', 'SYNTHETIC_RAW_'))
        assert 'observation.input' not in serialized and 'observation.output' not in serialized
    finally:
        sdk.shutdown()
        server.shutdown()
        server.server_close()
