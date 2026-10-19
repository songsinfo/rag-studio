"""Public installation must not silently reuse shared credentials."""
import pytest
import asyncio

from app.config import load_settings
from app.seed import DemoSeed


def test_unconfigured_gateway_has_no_shared_key(monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    assert load_settings().LITELLM_MASTER_KEY == ""


def test_unconfigured_csrf_has_no_shared_secret(monkeypatch):
    monkeypatch.delenv("RAG_CSRF_SECRET", raising=False)
    assert load_settings().RAG_CSRF_SECRET == ""


def test_seed_rejects_missing_password_before_database_access(monkeypatch):
    monkeypatch.delenv("RAG_SEED_PASSWORD", raising=False)
    monkeypatch.setattr("app.seed.metadata.create_all", lambda **kwargs: pytest.fail("database accessed before credential validation"))
    with pytest.raises(RuntimeError, match="RAG_SEED_PASSWORD"):
        DemoSeed.run()


@pytest.mark.parametrize("missing", ["RAG_CSRF_SECRET", "LITELLM_MASTER_KEY"])
def test_startup_rejects_missing_secrets_before_database_access(monkeypatch, missing):
    import app.main as main

    monkeypatch.setattr(main, "is_test_env", lambda: False)
    monkeypatch.setattr(main.settings, "RAG_CSRF_SECRET", "synthetic-csrf-for-test")
    monkeypatch.setattr(main.settings, "LITELLM_MASTER_KEY", "synthetic-gateway-for-test")
    monkeypatch.setattr(main.settings, missing, "")
    monkeypatch.setattr(main, "get_db_connection", lambda: pytest.fail("database accessed before credential validation"))

    async def start():
        async with main.lifespan(main.app):
            pytest.fail("startup accepted missing credentials")

    with pytest.raises(RuntimeError, match=missing):
        asyncio.run(start())
