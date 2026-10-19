from types import SimpleNamespace

import httpx
import pytest
from app.qa.model_gateway import ModelGateway


@pytest.mark.parametrize("usage_payload,expected_status,expected_tokens", [
    ({"usage": {"total_tokens": 178}}, "SUCCESS", 178),
    ({"meta": {"tokens": {"input_tokens": 178}}}, "SUCCESS", 178),
    ({"meta": {"tokens": {"input_tokens": 0}}}, "SUCCESS", 0),
    ({"meta": None}, "UNKNOWN", None),
    ({}, "UNKNOWN", None),
])
def test_rerank_preserves_gateway_usage_and_result_mapping(monkeypatch, usage_payload,
                                                         expected_status, expected_tokens):
    import app.qa.model_gateway as gateway

    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-rerank-key")
    requests = []
    calls = []
    monkeypatch.setattr(ModelGateway, "_record_call", lambda *args, **kwargs: calls.append(args))

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return httpx.Response(200, json={
            "results": [{"index": 1, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.1}],
            **usage_payload,
        })
    monkeypatch.setattr(gateway, "get_http_client", lambda: SimpleNamespace(post=post))
    result = ModelGateway.rerank("tenant-demo-1", None, "旅行人数", ["无关文本", "两大一小"], 2)
    assert requests[0][0] == "http://gateway.test/v1/rerank"
    assert requests[0][1]["model"] == "rerank-main"
    assert result[0] == {"index": 1, "relevance_score": 0.95}
    assert calls[0][4] == expected_status
    if expected_tokens is not None:
        assert calls[0][5] == expected_tokens
