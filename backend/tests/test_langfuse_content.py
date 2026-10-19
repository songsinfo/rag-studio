"""Optional bodies, real SDK serialization and synthetic Worker integration."""
import json
import uuid
from types import SimpleNamespace

import pytest
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetry
from test_evaluation import eval_env


@pytest.fixture
def observed(monkeypatch):
    provider = TracerProvider(shutdown_on_exit=False)
    exported, otlp = InMemorySpanExporter(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(telemetry._MetadataOnlyExporter(otlp)))
    sdk = Langfuse(public_key='pk-lf-content-' + uuid.uuid4().hex, secret_key='sk-lf-content-test',
        base_url='http://127.0.0.1:1', tracer_provider=provider, span_exporter=exported,
        should_export_span=telemetry._export_span, mask_otel_spans=telemetry._mask_otel_spans)
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    monkeypatch.setattr(telemetry, '_client', sdk)
    monkeypatch.setattr(telemetry, '_status', 'ENABLED')
    monkeypatch.setenv('RAG_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('RAG_LANGFUSE_SAMPLE_RATE', '1')
    monkeypatch.setenv('RAG_LANGFUSE_ID_HMAC_KEY', 'x'*40)
    for category in ('MODEL', 'RETRIEVAL', 'TOOL', 'QA'):
        monkeypatch.delenv('RAG_LANGFUSE_CAPTURE_' + category, raising=False)
    yield sdk, exported, otlp
    sdk.shutdown()
    provider.shutdown()


@pytest.mark.parametrize('mode', ['METADATA_ONLY', 'CONTENT', 'INVALID'])
def test_sdk_bodies_and_otlp_isolation(monkeypatch, observed, mode):
    sdk, exported, otlp = observed
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', mode)
    request = {'messages': [{'role': 'system', 'content': '完整合成提示'},
        {'role': 'tool', 'content': '{"evidence":"合成上下文"}'}], 'tools': [{'name': 'synthetic'}]}
    response = {'content': '', 'tool_calls': [{'name': 'read_section', 'args': {'sectionId': 's'}}]}
    with telemetry.observation_batch({'kind': 'QA_RUN', 'tenant_id': 't', 'target_id': 'r'}):
        telemetry.set_span_io('QA', input={'question': '合成问题'}, output={'answer': '合成答案'})
        with telemetry.model_observation('AGENT', 'chat-main', 'c') as observation:
            telemetry.set_model_io(observation, input=request, output=response)
            telemetry.finish_model_observation(observation, status='SUCCESS', duration_ms=1)
    sdk.flush()
    root = next(s for s in exported.get_finished_spans() if s.name == 'qa.worker.attempt')
    model = next(s for s in exported.get_finished_spans() if s.name == 'model.agent')
    if mode == 'CONTENT':
        assert json.loads(root.attributes['langfuse.observation.input']) == {'question': '合成问题'}
        assert json.loads(model.attributes['langfuse.observation.input']) == request
        assert json.loads(model.attributes['langfuse.observation.output']) == response
    else:
        assert all(not telemetry._IO_KEYS.intersection(s.attributes) for s in (root, model))
    assert all(not telemetry._IO_KEYS.intersection(s.attributes) for s in otlp.get_finished_spans())
    assert '完整合成提示' not in str([dict(s.attributes) for s in otlp.get_finished_spans()])


@pytest.mark.parametrize('category,name', [('MODEL','agent.model'), ('TOOL','agent.tool'),
    ('RETRIEVAL','rag.retrieve'), ('QA','qa.worker.attempt')])
def test_category_switch_removes_already_attached_content(monkeypatch, observed, category, name):
    from app.observability import agent_span
    sdk, exported, _ = observed
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT')
    with telemetry.observation_batch({'kind':'QA_RUN'}):
        with agent_span(name) as span:
            telemetry.set_span_io(category, span=span, input='SYNTHETIC_BODY')
            monkeypatch.setenv('RAG_LANGFUSE_CAPTURE_' + category, 'false')
    sdk.flush()
    assert all(not telemetry._IO_KEYS.intersection(s.attributes) for s in exported.get_finished_spans())


def test_large_or_unserializable_content_and_unsampled_trace(monkeypatch, observed):
    sdk, exported, _ = observed
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT')
    monkeypatch.setattr(telemetry, '_IO_MAX_BYTES', 128)
    with telemetry.observation_batch({'kind':'QA_RUN'}):
        telemetry.set_span_io('QA', input='中'*200, output=object())
    sdk.flush()
    root = exported.get_finished_spans()[0]
    marker = json.loads(root.attributes['langfuse.observation.input'])
    assert marker['captureStatus'] == 'OMITTED_TOO_LARGE' and marker['originalBytes'] > marker['limitBytes']
    assert 'langfuse.observation.output' not in root.attributes
    exported.clear()
    monkeypatch.setenv('RAG_LANGFUSE_SAMPLE_RATE', '0')
    with telemetry.observation_batch({'kind':'QA_RUN'}):
        telemetry.set_span_io('QA', input='NOT_SAMPLED')
    sdk.flush()
    assert not exported.get_finished_spans()


def test_response_projection_and_capture_failure_do_not_change_business(monkeypatch):
    from langchain_core.messages import AIMessage
    message = AIMessage(content='', tool_calls=[{'id':'c','name':'read_section','args':{'sectionId':'s'}}],
        response_metadata={'authorization':'SECRET_HEADER'})
    assert telemetry.model_response_content(message)['tool_calls'][0]['args'] == {'sectionId':'s'}
    assert 'SECRET_HEADER' not in str(telemetry.model_response_content(message))
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT')
    monkeypatch.setattr(telemetry, 'export_current_span', lambda: True)
    called = []
    failing = SimpleNamespace(update=lambda **_: (_ for _ in ()).throw(RuntimeError('SECRET_BODY')))
    telemetry.set_model_io(failing, input={'messages': []})
    called.append('business')
    assert called == ['business']


@pytest.mark.parametrize('mode', ['METADATA_ONLY', 'CONTENT'])
def test_worker_model_retrieval_mock_tool_and_root_bodies(monkeypatch, eval_env, observed, mode):
    from sqlalchemy import select
    from app.db import get_db_connection
    from app.models import config_revision_table, run_event_table
    from app.qa.model_gateway import ModelGateway
    from app.qa.model_usage import invoke_tracked
    from app.qa.service import QaService
    from app.runtime.worker import DurableJobWorker
    import app.qa.service as service
    sdk, exported, _ = observed
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', mode)
    monkeypatch.setattr(service, 'is_test_env', lambda: False)
    actor = eval_env['admin']
    with get_db_connection() as conn:
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
            id='cfg-model-chat-main', kind='MODEL', name='synthetic', revision=1, sha256=b'0'*32,
            status='PUBLISHED', payload={'modelAlias':'chat-main','currency':'CNY',
                'inputPricePerMillion':1,'outputPricePerMillion':1}))
        conn.commit()
    sent = []
    def synthetic(*, tenant_id, run_id, purpose, messages, tools, **kwargs):
        tool_messages = [m for m in messages if m.get('role') == 'tool']
        if not tool_messages:
            reply = {'role':'assistant','content':'','tool_calls':[
                {'id':'search','name':'search_knowledge','args':{'query':'知识库系统采用什么分块策略？','topK':5}},
                {'id':'mock','name':'get_final_price_rules','args':{}}]}
        else:
            evidence = json.loads(tool_messages[0]['content'])['evidence'][0]
            reply = {'role':'assistant','tool_calls':[], 'content':json.dumps({'status':'COMPLETED',
                'claims':[{'id':'c1','text':evidence['excerpt'][:10],
                    'evidence':[{'evidenceId':evidence['evidenceId'],'start':0,'end':10}]}],
                'gaps':[],'conflicts':[]})}
        sent.append((json.loads(json.dumps({'messages':messages,'tools':tools})), reply))
        return invoke_tracked(lambda: reply, tenant_id=tenant_id, run_id=run_id,
            purpose=purpose, alias='chat-main', payload={'messages':messages,'tools':tools},
            max_output_tokens=20, usage_reader=lambda _: {'input_tokens':30,'output_tokens':10})
    monkeypatch.setattr(ModelGateway, 'chat_tools', synthetic)
    conversation = QaService.create_conversation(actor, {'title':'synthetic content'})
    run_id = QaService.submit_run(actor, conversation['conversationId'],
        {'question':'知识库系统采用什么分块策略？','segmentId':conversation['currentSegmentId']},
        idempotency_key='content-worker', trusted_config_id='cfg-enterprise-retrieval-v2')['runId']
    worker = DurableJobWorker(kinds={'QA_RUN'})
    assert worker.run_once()
    detail = QaService.get_run(actor, run_id)
    assert detail['status'] == 'COMPLETED' and len(sent) == 2 and detail['mock'] is True
    sdk.flush()
    spans = list(exported.get_finished_spans())
    required = {'qa.worker.attempt','agent.run','agent.model','agent.tool','rag.retrieve',
        'rag.prepare_query','rag.bm25_recall','rag.context_budget','rag.finalize_hits','model.agent'}
    assert required.issubset({s.name for s in spans})
    if mode == 'METADATA_ONLY':
        assert all(not telemetry._IO_KEYS.intersection(s.attributes) for s in spans)
    else:
        assert all(telemetry._IO_KEYS.issubset(s.attributes) for s in spans if s.name in required)
        generations = [s for s in spans if s.name == 'model.agent']
        assert [json.loads(s.attributes['langfuse.observation.input']) for s in generations] == [p for p,r in sent]
        assert [json.loads(s.attributes['langfuse.observation.output']) for s in generations] == [r for p,r in sent]
        tool = next(s for s in spans if s.name == 'agent.tool' and s.attributes.get('tool.mock') is True)
        assert json.loads(tool.attributes['langfuse.observation.output'])['mock'] is True
        root = next(s for s in spans if s.name == 'qa.worker.attempt')
        assert json.loads(root.attributes['langfuse.observation.output'])['answer'] == detail['answer']
        retrieval = next(s for s in spans if s.name == 'rag.finalize_hits')
        assert eval_env['quote'] in str(json.loads(retrieval.attributes['langfuse.observation.output']))
    with get_db_connection() as conn:
        events = conn.execute(select(run_event_table.c.payload).where(run_event_table.c.run_id == run_id)).scalars().all()
    assert not any('langfuse.observation.input' in str(event) for event in events)


def test_revoked_retrieval_content_is_skipped_without_changing_result(monkeypatch, eval_env, observed):
    from app.retrieval.rag_graph import RagRetrievalGraph, RagRetrievalContext
    from app.observability import agent_span
    import app.retrieval.rag_graph as graph
    from app.errors import ApiException
    sdk, exported, _ = observed
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT')
    ctx = RagRetrievalContext(eval_env['admin'], 'synthetic', None, None, None, None, None, None, [], None)
    data = {'hits':[{'text':'REVOKED_CANARY'}]}
    monkeypatch.setattr(graph, 'assert_hits_still_readable', lambda *a: (_ for _ in ()).throw(
        ApiException(409, 'SOURCE_CHANGED', 'revoked')))
    with telemetry.observation_batch({'kind':'QA_RUN'}):
        with agent_span('rag.rerank') as span:
            RagRetrievalGraph._capture_io(ctx, data, span, 'output')
    sdk.flush()
    assert 'REVOKED_CANARY' not in str([dict(s.attributes) for s in exported.get_finished_spans()])
    assert data == {'hits':[{'text':'REVOKED_CANARY'}]}


def test_source_change_during_model_call_skips_body_without_repeating_call(monkeypatch, eval_env, observed):
    from sqlalchemy import select
    from app.db import get_db_connection
    from app.errors import ApiException
    from app.models import config_revision_table, job_table, model_call_table
    from app.qa.model_gateway import ModelGateway
    from app.qa.model_usage import invoke_tracked
    sdk, exported, _ = observed
    actor = eval_env['admin']
    monkeypatch.setenv('RAG_LANGFUSE_DATA_MODE', 'CONTENT')
    with get_db_connection() as conn:
        conn.execute(config_revision_table.insert().values(tenant_id=actor.tenant_id,
            id='cfg-model-chat-main',kind='MODEL',name='synthetic',revision=1,sha256=b'0'*32,
            status='PUBLISHED',payload={'modelAlias':'chat-main','currency':'CNY',
                'inputPricePerMillion':1,'outputPricePerMillion':1}))
        conn.execute(job_table.insert().values(tenant_id=actor.tenant_id,id='body-revoked',
            kind='EVALUATE',target_id='body-revoked',requested_by=actor.user_id,
            idempotency_key='body-revoked',request_hash=b'0'*32,status='RUNNING'))
        conn.commit()
    checks, calls = [], []
    def check(*args, **kwargs):
        checks.append(True)
        if len(checks) > 1:
            raise ApiException(409,'SOURCE_CHANGED','revoked after sending')
    monkeypatch.setattr(ModelGateway,'assert_run_sources_current',check)
    reply={'content':'POST_REVOCATION_RESPONSE'}
    with telemetry.observation_batch({'kind':'EVALUATE'}):
        actual=invoke_tracked(lambda: (calls.append('sent') or reply),tenant_id=actor.tenant_id,
            job_id='body-revoked',purpose='ANSWER',alias='chat-main',payload={'messages':[]},
            max_output_tokens=20,usage_reader=lambda _: {'input_tokens':10,'output_tokens':2})
    sdk.flush()
    assert actual == reply and calls == ['sent']
    generation=next(s for s in exported.get_finished_spans() if s.name=='model.answer')
    assert 'langfuse.observation.output' not in generation.attributes
    with get_db_connection() as conn:
        rows=conn.execute(select(model_call_table).where(model_call_table.c.job_id=='body-revoked')).all()
    assert len(rows)==1 and rows[0].status=='SETTLED'
