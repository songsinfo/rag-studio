#!/usr/bin/env python3
"""Native developer entry point. For the full stack use quickstart.py instead.

Examples:
  python scripts/dev.py init --environment demo
  python scripts/dev.py deps --environment demo --documents
  python scripts/dev.py backend --environment demo --port 8080
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from quickstart import ENVIRONMENTS, ROOT, env_path, initialize, read_env, run_docker
from tooling import CheckError, load_environment, log_error, python_executable


def start_dependencies(environment: str, documents: bool = False) -> None:
    if not env_path(environment).exists():
        initialize(environment)
    values = read_env(env_path(environment))
    command = [
        "docker", "compose", "--project-name", f"rag-{environment}",
        "--env-file", str(env_path(environment)),
        "-f", str(ROOT / "deploy" / "compose.yml"),
    ]
    if documents:
        command += ["--profile", "documents"]
    if values.get("RAG_ARTIFACT_STORE", "").lower() == "s3":
        command += ["--profile", "artifacts-s3"]
    # Reuse Quick Start's isolated Compose environment handling.
    run_docker(command + ["up", "-d", "--wait", "--wait-timeout", "240"], values)


def start_backend(environment: str, port: int) -> None:
    if not (1 <= port <= 65535):
        raise CheckError("Port must be between 1 and 65535")
    env = load_environment(environment)
    runtime = ROOT / ".runtime" / environment
    for directory in ("artifacts", "import", "logs"):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    env.update({
        "PORT": str(port),
        "RAG_ARTIFACT_ROOT": str(runtime / "artifacts"),
        "RAG_IMPORT_ROOT": str(runtime / "import"),
        "RAG_LOG_FILE": str(runtime / "logs" / "backend.log"),
        "PYTHONPATH": str(ROOT / "backend"),
    })
    executable = python_executable()
    subprocess.run(
        [executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT / "backend", env=env, check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("init", "deps", "backend"):
        p = sub.add_parser(action)
        p.add_argument("--environment", choices=ENVIRONMENTS, default="demo")
        if action == "deps":
            p.add_argument("--documents", action="store_true")
        if action == "backend":
            p.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            initialize(args.environment)
        elif args.action == "deps":
            start_dependencies(args.environment, documents=args.documents)
        else:
            start_backend(args.environment, args.port)
    except (OSError, CheckError, ValueError, subprocess.CalledProcessError) as exc:
        return log_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
