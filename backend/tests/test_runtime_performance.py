import os

from app.qa.agent import LangGraphAgent
from app.retrieval import embedding


def test_agent_graph_is_compiled_once():
    LangGraphAgent._graph = None
    first = LangGraphAgent._get_graph()
    second = LangGraphAgent._get_graph()
    assert first is second


def test_embedding_batches_large_requests(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway/v1")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")

    calls = []

    class Response:
        status_code = 200
        text = ""
        def __init__(self, batch):
            self.batch = batch
        def json(self):
            return {
                "data": [
                    {"index": i, "embedding": [float(i + 1)] * 4}
                    for i, _ in enumerate(self.batch)
                ]
            }

    class Client:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(list(json["input"]))
            return Response(json["input"])

    monkeypatch.setattr(embedding, "get_http_client", lambda: Client())
    result = embedding.get_embeddings(["a", "b", "c", "d", "e"], dim=4)
    assert len(result) == 5
    assert [len(batch) for batch in calls] == [2, 2, 1]


def test_langchain_v1_chat_model_and_tool_binding(monkeypatch):
    from app.qa.agent import AGENT_TOOLS
    from app.qa.model_clients import _build_chat_model, get_chat_model

    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-key")
    _build_chat_model.cache_clear()

    model = get_chat_model("chat-main", 512)
    assert model is not None
    bound = model.bind_tools(AGENT_TOOLS, strict=True)
    assert bound is not None


def test_agent_runtime_policy_is_configurable(monkeypatch):
    from app.qa.runtime import AgentRuntimePolicy

    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "4")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "5")
    monkeypatch.setenv("AGENT_MAX_CONTEXT_TOKENS", "9000")
    policy = AgentRuntimePolicy.from_env()
    assert policy.overall_timeout_seconds == 45
    assert policy.max_rounds == 4
    assert policy.max_tool_calls == 5
    assert policy.max_context_tokens == 9000
