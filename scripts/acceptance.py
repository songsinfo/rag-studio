#!/usr/bin/env python3
"""Opt-in local backend, LiteLLM and paid-model checks.

All network/paid tests require an explicit command. CI only runs --self-test.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

from quickstart import ROOT, read_env, env_path
from tooling import (CheckError, HttpClient, assert_that, assert_http,
                     log_error, python_executable, seed_password)


def backend_contract(base: str, environment: str) -> None:
    client = HttpClient(base)
    csrf = client.ok("GET", "/api/v1/auth/csrf")
    token = csrf["csrfToken"]
    assert_that(bool(token), "Missing CSRF token")
    no_csrf = client.request("POST", "/api/v1/auth/login",
                             {"tenantCode": "demo", "loginName": "admin",
                              "password": seed_password(environment)})
    assert_that(no_csrf.status in (400, 403), "Missing CSRF not rejected")
    bad = client.request("POST", "/api/v1/auth/login",
                         {"tenantCode": "demo", "loginName": "admin", "password": "wrong-password"},
                         {"X-CSRF-Token": token})
    assert_that(bad.status in (400, 401), "Bad login not rejected")
    client.login(environment, "admin", seed_password(environment))
    me = client.ok("GET", "/api/v1/me")
    assert_that(me.get("admin") is True and me.get("loginName") == "admin", "Admin identity mismatch")
    assert_http(client.request("GET", "/api/v1/knowledge-bases"), 200, label="list knowledge bases")
    kb = client.ok("POST", "/api/v1/admin/knowledge-bases",
                   {"name": "contract-kb-" + uuid.uuid4().hex[:8],
                    "readers": [{"type": "USER", "userId": me["userId"]}]},
                   client.headers())
    kid = kb["id"]
    acl = client.ok("GET", f"/api/v1/admin/knowledge-bases/{kid}/acl")
    assert_that(acl.get("rowVersion") is not None, "Missing ACL rowVersion")
    conflict = client.request("PUT", f"/api/v1/admin/knowledge-bases/{kid}/acl",
                              {"expectedVersion": 999999,
                               "readers": [{"type": "USER", "userId": me["userId"]}]},
                              client.headers())
    assert_that(conflict.status in (409, 412), "Optimistic-lock conflict not rejected")
    assert_http(client.request("GET", f"/api/v1/documents?kbId={kid}"), 200, label="documents")
    search = client.ok("GET", f"/api/v1/search?query=test&kbIds={kid}")
    assert_that(search.get("evidence") is not None, "Missing search evidence")
    session = client.request("POST", "/api/v1/sessions", {"kbIds": [kid], "mode": "RAG"}, client.headers())
    assert_that(session.status in (200, 201) and (session.body or {}).get("id"), "QA session creation failed")
    logout = client.request("POST", "/api/v1/auth/logout", {}, client.headers(idempotent=False))
    assert_that(logout.status in (200, 204), "Logout failed")
    assert_that(client.request("GET", "/api/v1/me").status in (401, 403),
                "Session was not revoked")
    print("Backend API contract checks passed")


def gateway(base: str, master_key: str, environment: str) -> None:
    assert_that(bool(master_key), "Set LITELLM_MASTER_KEY or --master-key")
    client = HttpClient(base)
    health = client.request("GET", "/health/liveliness")
    if health.status != 200:
        health = client.request("GET", "/health")
    assert_http(health, 200, label="LiteLLM health")
    assert_that(client.request("GET", "/models").status in (401, 403),
                "Unauthenticated /models request allowed")
    auth = {"Authorization": "Bearer " + master_key}
    models = client.request("GET", "/models", headers=auth)
    assert_http(models, 200, label="models")
    names = {row.get("id") for row in (models.body or {}).get("data", [])}
    assert_that({"chat-main", "embedding-main", "judge-main"} <= names, "Missing stable aliases")
    bad = client.request("POST", "/chat/completions",
                         {"model": "non-existent-model", "messages": [{"role": "user", "content": "ping"}]},
                         auth)
    assert_that(bad.status != 200 and isinstance(bad.body, dict) and bad.body.get("error") is not None,
                "Invalid model did not yield compatible error")
    path = env_path(environment)
    if path.is_file():
        upstream = read_env(path).get("MODEL_API_KEY", "")
        assert_that(not (len(upstream) > 8 and upstream in bad.raw), "Upstream API key leaked")
    rerank = client.request("POST", "/rerank",
                            {"model": "rerank-main", "query": "test", "documents": ["doc1"]}, auth)
    assert_that(rerank.status in (400, 404, 405, 501),
                "Rerank enabled without a locked contract")
    chat = client.request("POST", "/chat/completions",
                          {"model": "chat-main", "messages": [{"role": "user", "content": "Respond with exact word OK"}],
                           "max_tokens": 10}, auth)
    assert_http(chat, 200, label="chat completion")
    assert_that(bool(chat.body.get("choices")) and
                chat.body.get("usage", {}).get("prompt_tokens") is not None,
                "Chat completion contract missing choices/usage")
    embed = client.request("POST", "/embeddings", {"model": "embedding-main", "input": "ping"}, auth)
    assert_http(embed, 200, label="embeddings")
    assert_that(bool(embed.body.get("data")) and bool(embed.body["data"][0].get("embedding")),
                "Embedding array empty")
    print("LiteLLM Gateway checks passed")


def real_acceptance(kind: str, cap: float) -> None:
    if not 0.01 <= cap <= 5:
        raise CheckError("Max cost CNY must be 0.01..5")
    tests = {"model": ("RAG_REAL_MODEL_ACCEPTANCE", "RAG_REAL_MODEL_MAX_COST_CNY",
                       "backend/tests/test_real_model_acceptance.py", ".local/pytest-real-model"),
             "assertions": ("RAG_REAL_ASSERTION_ACCEPTANCE", "RAG_REAL_ASSERTION_MAX_COST_CNY",
                            "backend/tests/test_real_assertion_acceptance.py", ".local/pytest-real-assertion")}
    flag, budget, test_path, tmp = tests[kind]
    env = {**os.environ, flag: "1", budget: str(cap)}
    (ROOT / ".local").mkdir(exist_ok=True)
    subprocess.run([python_executable(), "-m", "pytest", test_path, "-q", "-p", "no:cacheprovider",
                    "--tb=short", "--basetemp", tmp], cwd=ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("backend", "gateway"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:" + ("8080" if name == "backend" else "4000"))
        p.add_argument("--environment", choices=("demo", "evaluation", "test"), default="demo")
        p.add_argument("--self-test", action="store_true")
        if name == "gateway":
            p.add_argument("--master-key", default="")
    p = sub.add_parser("real")
    p.add_argument("kind", choices=("model", "assertions"))
    p.add_argument("--max-cost-cny", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action in ("backend", "gateway") and args.self_test:
            assert_http(__import__("tooling").ApiResponse(409, {"code": "EXPECTED"}, ""), 409, "EXPECTED")
            print("HTTP contract assertion self-test passed")
            return 0
        if args.action == "backend":
            backend_contract(args.base_url, args.environment)
        elif args.action == "gateway":
            key = args.master_key
            if not key and env_path(args.environment).is_file():
                key = read_env(env_path(args.environment)).get("LITELLM_MASTER_KEY", "")
            gateway(args.base_url, key, args.environment)
        else:
            real_acceptance(args.kind, args.max_cost_cny)
    except (CheckError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        return log_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
