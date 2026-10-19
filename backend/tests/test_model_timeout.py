import time
from types import SimpleNamespace

import httpx
import pytest

from app.qa.model_gateway import ModelGateway


@pytest.mark.parametrize("remaining", [None, 10])
@pytest.mark.parametrize("purpose", ["ANSWER", "JUDGE"])
def test_answer_and_judge_wait_for_slow_model_without_exceeding_deadline(monkeypatch, remaining, purpose):
    import app.qa.model_gateway as gateway

    monkeypatch.setenv("LITELLM_BASE_URL", "http://model.test")
    monkeypatch.setenv("AGENT_CALL_MAX_ATTEMPTS", "1")
    deadline = time.time() + remaining if remaining is not None else None

    def fake_model(alias, max_tokens, timeout_seconds):
        def invoke(messages):
            if remaining is None and timeout_seconds < 40:
                raise httpx.ReadTimeout("A valid answer needs 40 seconds")
            if deadline is not None:
                assert 0 < timeout_seconds <= remaining
            return SimpleNamespace(content="回答", usage_metadata={"input_tokens": 10, "output_tokens": 2})
        model = SimpleNamespace(invoke=invoke)
        model.bind = lambda **kwargs: model
        return model

    monkeypatch.setattr(gateway, "get_chat_model", fake_model)
    monkeypatch.setattr(gateway, "invoke_tracked", lambda fn, **kwargs: fn())
    assert ModelGateway.chat(
        "tenant-demo-1", None, purpose, [{"role": "user", "content": "问题"}],
        deadline_epoch=deadline,
    ) == "回答"


@pytest.mark.parametrize("purpose", ["ANSWER", "HYDE"])
def test_answer_json_mode_preserves_plain_text_retrieval_calls(monkeypatch, purpose):
    import json
    import app.qa.model_gateway as gateway

    monkeypatch.setenv("LITELLM_BASE_URL", "http://model.test")
    monkeypatch.setattr(gateway, "invoke_tracked", lambda fn, **kwargs: fn())
    options = {}

    def invoke(messages):
        content = '{"claims":[]}' if options.get("response_format") == {"type": "json_object"} else ('```json\n{"claims":[]}\n```' if purpose == "ANSWER" else "hypothetical passage")
        return SimpleNamespace(content=content)

    model = SimpleNamespace(invoke=invoke)
    def bind(**kwargs):
        options.update(kwargs)
        return model
    model.bind = bind
    monkeypatch.setattr(gateway, "get_chat_model", lambda *args: model)
    result = ModelGateway.chat("tenant-demo-1", None, purpose, [{"role": "user", "content": "JSON 问题"}])
    if purpose == "ANSWER":
        assert json.loads(result) == {"claims": []}
    else:
        assert result == "hypothetical passage"
        assert not options


@pytest.mark.parametrize("requested, expected", [(60, 60), (45, 45), (30, 30), (90, 60)])
def test_chat_model_client_preserves_answer_timeout(monkeypatch, requested, expected):
    import app.qa.model_clients as clients

    monkeypatch.setenv("LITELLM_BASE_URL", "http://model.test")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-key")
    captured = {}

    def fake_build(base_url, api_key, model_alias, max_tokens, timeout_seconds):
        captured["timeout"] = timeout_seconds
        return object()

    monkeypatch.setattr(clients, "_build_chat_model", fake_build)
    assert clients.get_chat_model("chat-main", 512, requested) is not None
    assert captured["timeout"] == expected
