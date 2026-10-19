import atexit
import os
import threading
from functools import lru_cache
from typing import Optional

import httpx
from langchain_openai import ChatOpenAI


_http_lock = threading.Lock()
_http_client: Optional[httpx.Client] = None


def _base_url() -> str:
    raw = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
    if not raw:
        return ""
    return raw if raw.endswith("/v1") else f"{raw}/v1"


@lru_cache(maxsize=32)
def _build_chat_model(
    base_url: str,
    api_key: str,
    model_alias: str,
    max_tokens: int,
    timeout_seconds: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_alias,
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=timeout_seconds,
        # LiteLLM 作为 OpenAI-compatible Chat Completions 网关使用；具体供应商差异统一在网关层吸收。
        # 这里显式固定 API 形态，避免仅凭模型名启发式判断而静默切换协议，导致请求结构不兼容。
        use_responses_api=False,
    )


def get_chat_model(
    model_alias: str,
    max_tokens: int,
    timeout_seconds: float = 30.0,
) -> Optional[ChatOpenAI]:
    base_url = _base_url()
    if not base_url:
        return None
    api_key = os.getenv("LITELLM_MASTER_KEY", "")
    # ANSWER calls may use up to 60 seconds; other purposes are bounded by their caller.
    timeout_seconds = max(1.0, min(60.0, float(timeout_seconds)))
    return _build_chat_model(base_url, api_key, model_alias, max_tokens, timeout_seconds)


def get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is not None:
        return _http_client
    with _http_lock:
        if _http_client is None:
            _http_client = httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=False,
            )
    return _http_client


def close_clients() -> None:
    global _http_client
    with _http_lock:
        if _http_client is not None:
            _http_client.close()
            _http_client = None
    _build_chat_model.cache_clear()


atexit.register(close_clients)
