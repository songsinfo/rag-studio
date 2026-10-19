#!/usr/bin/env python3
"""Private, synthetic Langfuse acceptance environment.

Do not use real customer data or share .local/langfuse-eval.env.
"""
from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path

from quickstart import ROOT, read_env, run_docker
from tooling import CheckError, log_error, python_executable

LOCAL_FILE = ROOT / ".local" / "langfuse-eval.env"
COMPOSE = ROOT / "deploy" / "langfuse.compose.yml"


def initialize() -> None:
    LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "LFD_POSTGRES_PASSWORD": secrets.token_hex(32),
        "LFD_CLICKHOUSE_PASSWORD": secrets.token_hex(32),
        "LFD_MINIO_PASSWORD": secrets.token_hex(32),
        "LFD_REDIS_PASSWORD": secrets.token_hex(32),
        "LFD_AUTH_SECRET": secrets.token_hex(32),
        "LFD_SALT": secrets.token_hex(32),
        "LFD_ENCRYPTION_KEY": secrets.token_hex(32),
        "LFD_USER_PASSWORD": secrets.token_hex(32),
        "LANGFUSE_PUBLIC_KEY": "pk-lf-" + str(uuid.uuid4()),
        "LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(32),
        "LANGFUSE_BASE_URL": "http://127.0.0.1:3031",
        "RAG_LANGFUSE_ID_HMAC_KEY": secrets.token_hex(32),
        "RAG_LANGFUSE_ENABLED": "true",
        "RAG_LANGFUSE_SAMPLE_RATE": "1",
        "RAG_LANGFUSE_TARGET_ID": "local-synthetic",
    }
    content = "".join(f"{key}={value}\n" for key, value in values.items())
    try:
        fd = os.open(LOCAL_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CheckError("Local Langfuse secrets already exist; reuse that file") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"Created local-only Langfuse credentials: {LOCAL_FILE}")


def command() -> list[str]:
    if not LOCAL_FILE.is_file():
        raise CheckError("Run: python scripts/langfuse_local.py init")
    return [
        "docker", "compose", "--env-file", str(LOCAL_FILE),
        "-f", str(COMPOSE),
    ]


def run_compose(*args: str) -> None:
    run_docker(command() + list(args), read_env(LOCAL_FILE))


def check(*, outage: bool, content: bool) -> None:
    values = read_env(LOCAL_FILE) if LOCAL_FILE.is_file() else None
    if values is None:
        raise CheckError("Run: python scripts/langfuse_local.py init")
    env = os.environ.copy()
    env.update({k: v for k, v in values.items() if k.startswith(("LANGFUSE_", "RAG_LANGFUSE_"))})
    env.update({
        "RAG_LANGFUSE_PRIVATE_ACCEPTANCE": "1",
        "RAG_LANGFUSE_FAULT_ACCEPTANCE": "1" if outage else "0",
        "RAG_LANGFUSE_CONTENT_ACCEPTANCE": "1" if content else "0",
    })
    # Fault acceptance has an outage window; always attempt service restoration.
    try:
        if outage:
            run_compose("stop", "web", "worker")
        subprocess.run([
            python_executable(), "-m", "pytest", "backend/tests/test_langfuse_private.py",
            "-q", "-p", "no:cacheprovider", "--tb=short",
            "--basetemp", ".local/pytest-langfuse-acceptance",
        ], cwd=ROOT, env=env, check=True)
    finally:
        if outage:
            run_compose("start", "web", "worker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    sub.add_parser("up")
    sub.add_parser("down")
    p = sub.add_parser("check")
    p.add_argument("--outage", action="store_true")
    p.add_argument("--content", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            initialize()
        elif args.action == "up":
            run_compose("up", "-d")
        elif args.action == "down":
            run_compose("down")
        else:
            check(outage=args.outage, content=args.content)
    except (OSError, CheckError, ValueError, subprocess.CalledProcessError) as exc:
        return log_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
