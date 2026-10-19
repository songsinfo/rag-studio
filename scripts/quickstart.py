#!/usr/bin/env python3
"""Cross-platform Docker Compose Quick Start for the local RAG Studio demo.

No third-party Python packages are needed. Never prints generated credentials.
"""
from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENTS = {"demo": 0, "evaluation": 1, "test": 2}
SECRET_NAMES = (
    "DB_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "RAG_SEED_PASSWORD",
    "RAG_CSRF_SECRET",
    "DOCLING_API_KEY",
    "S3_SECRET_ACCESS_KEY",
)
REQUIRED_SETTINGS = (
    "DB_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "RAG_SEED_PASSWORD",
    "RAG_CSRF_SECRET",
    "DOCLING_API_KEY",
    "LITELLM_MASTER_KEY",
    "MODEL_BASE_URL",
    "MODEL_API_KEY",
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "JUDGE_MODEL",
)


def env_path(environment: str) -> Path:
    return ROOT / ".runtime" / (environment + ".env")


def read_env(path: Path) -> dict[str, str]:
    """Read the simple KEY=value template without executing or logging it."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def build_environment(environment: str) -> str:
    """Render a new private config, keeping public template values intact."""
    offset = ENVIRONMENTS[environment]
    replacements = {
        "RAG_ENVIRONMENT": environment,
        "COMPOSE_PROJECT_NAME": "rag-" + environment,
        "DB_NAME": "rag_" + environment,
        "DB_PORT": str(3307 + offset),
        "MILVUS_PORT": str(19530 + offset),
        "MILVUS_HOST": "127.0.0.1",
        "MILVUS_URI": "http://localhost:" + str(19530 + offset),
        "DOCLING_PORT": str(5001 + offset),
        "DOCLING_URL": "http://localhost:" + str(5001 + offset),
        "LITELLM_PORT": str(4000 + offset),
        "LITELLM_BASE_URL": "http://localhost:" + str(4000 + offset),
        "RAG_WEB_PORT": str(5173 + offset),
        "DB_URL": (
            f"jdbc:mysql://localhost:{3307 + offset}/rag_{environment}"
            "?connectionTimeZone=UTC&forceConnectionTimeZoneToSession=true"
        ),
        "RAG_INDEX_PREFIX": "rag_" + environment,
        "LITELLM_MASTER_KEY": "sk-" + secrets.token_hex(24),
    }
    replacements.update({key: secrets.token_hex(24) for key in SECRET_NAMES})
    source = (ROOT / ".env.example").read_text(encoding="utf-8")
    rendered = []
    seen = set()
    for line in source.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in replacements:
            rendered.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            rendered.append(line)
    missing = sorted(set(replacements) - seen)
    if missing:
        raise ValueError("Missing keys in .env.example: " + ", ".join(missing))
    return "\n".join(rendered) + "\n"


def initialize(environment: str) -> None:
    target = env_path(environment)
    if target.exists():
        print(f"Configuration exists (unchanged): {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL avoids overwriting existing credentials even under concurrent init.
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(f"Configuration exists (unchanged): {target}")
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(build_environment(environment))
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    print(f"Created private configuration: {target}")
    print("Fill in MODEL_BASE_URL, MODEL_API_KEY, CHAT_MODEL, EMBEDDING_MODEL and JUDGE_MODEL before starting.")
    print("Credentials are not printed; keep .runtime/ out of version control.")


def check_configuration(environment: str) -> dict[str, str]:
    target = env_path(environment)
    if not target.is_file():
        raise ValueError(f"Missing {target}; run 'python scripts/quickstart.py init' first.")
    values = read_env(target)
    missing = [name for name in REQUIRED_SETTINGS if not values.get(name, "").strip()]
    if missing:
        raise ValueError(
            "Missing configuration in " + str(target) + ": " + ", ".join(missing)
            + ". Do not commit this file."
        )
    if not values["MODEL_BASE_URL"].startswith(("http://", "https://")):
        raise ValueError("MODEL_BASE_URL must begin with http:// or https://")
    return values


def compose_command(environment: str, values: dict[str, str]) -> list[str]:
    command = [
        "docker",
        "compose",
        "--project-name",
        values.get("COMPOSE_PROJECT_NAME") or "rag-" + environment,
        "--env-file",
        str(env_path(environment)),
        "-f",
        str(ROOT / "deploy" / "compose.yml"),
        "-f",
        str(ROOT / "deploy" / "app.compose.yml"),
        "--profile",
        "documents",
    ]
    if values.get("RAG_ARTIFACT_STORE", "").strip().lower() == "s3":
        command += ["--profile", "artifacts-s3"]
    return command


def run_docker(command: list[str], values: dict[str, str]) -> None:
    # Compose gives shell variables precedence over the selected --env-file.
    child_env = os.environ.copy()
    for name in values:
        child_env.pop(name, None)
    try:
        subprocess.run(command, cwd=ROOT, check=True, env=child_env)
    except FileNotFoundError as exc:
        raise ValueError("Docker is unavailable. Install Docker Desktop or Docker Engine with Compose v2.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG Studio cross-platform Docker Compose Quick Start")
    parser.add_argument("--environment", choices=tuple(ENVIRONMENTS), default="demo")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Generate private demo credentials without overwriting existing ones")
    commands.add_parser("config", help="Validate all Docker Compose files (no containers started)")
    commands.add_parser("up", help="Build and start the full demo stack")
    commands.add_parser("ps", help="Show demo containers")
    logs = commands.add_parser("logs", help="View container logs")
    logs.add_argument("--service", default="", help="Optional service, e.g. backend or docling")
    logs.add_argument("--follow", action="store_true")
    down = commands.add_parser("down", help="Stop containers; preserve named volumes by default")
    down.add_argument("--volumes", action="store_true", help="DANGER: also remove persistent database/index volumes")
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            initialize(args.environment)
            return 0

        target = env_path(args.environment)
        if not target.is_file():
            raise ValueError(f"Missing {target}; run 'python scripts/quickstart.py init' first.")
        values = check_configuration(args.environment) if args.command in ("up", "config") else read_env(target)
        base = compose_command(args.environment, values)
        if args.command == "config":
            run_docker(base + ["config", "--quiet"], values)
            print("Docker Compose configuration is valid.")
        elif args.command == "up":
            run_docker(base + ["config", "--quiet"], values)
            run_docker(base + ["up", "--detach", "--build", "--wait", "--wait-timeout", "600"], values)
            port = values.get("RAG_WEB_PORT") or "5173"
            print(f"RAG Studio: http://localhost:{port}")
            print("Login: tenant=demo, user=admin, password=RAG_SEED_PASSWORD in the private env file.")
        elif args.command == "ps":
            run_docker(base + ["ps"], values)
        elif args.command == "logs":
            run_docker(base + ["logs", "--tail", "100"] + (["--follow"] if args.follow else []) + ([args.service] if args.service else []), values)
        elif args.command == "down":
            run_docker(base + ["down"] + (["--volumes"] if args.volumes else []), values)
    except ValueError as exc:
        print("Quick Start error: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
