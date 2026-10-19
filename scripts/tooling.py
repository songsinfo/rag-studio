"""Shared, dependency-free local tooling for Linux, macOS and Windows.

Do not print credentials, send local demo secrets to external services, or load
untrusted environment files. All HTTP checks are opt-in.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPCookieProcessor, ProxyHandler, Request, build_opener

from quickstart import ENVIRONMENTS, ROOT, env_path, read_env


class CheckError(RuntimeError):
    """A failed local contract check without secret-bearing response dumps."""


def required_environment(environment: str) -> dict[str, str]:
    if environment not in ENVIRONMENTS:
        raise CheckError(f"Unknown environment: {environment}")
    path = env_path(environment)
    if not path.is_file():
        raise CheckError(f"Missing {path}. Run: python scripts/quickstart.py init")
    return read_env(path)


def seed_password(environment: str) -> str:
    password = required_environment(environment).get("RAG_SEED_PASSWORD", "").strip()
    if not password:
        raise CheckError(f"RAG_SEED_PASSWORD missing from .runtime/{environment}.env")
    return password


def python_executable() -> str:
    """Prefer installed backend venv on any OS; fall back to current Python."""
    base = ROOT / "backend" / ".venv"
    for child in ("bin/python", "Scripts/python.exe"):
        path = base / child
        if path.is_file():
            return str(path)
    return sys.executable


def load_environment(environment: str) -> dict[str, str]:
    """Process-only environment derived from the selected private file."""
    return {**os.environ, **required_environment(environment)}


def assert_that(condition: Any, message: str) -> None:
    if not condition:
        raise CheckError(message)


@dataclass
class ApiResponse:
    status: int
    body: Any
    raw: str


def assert_http(response: ApiResponse, status: int, code: str | None = None, label: str = "") -> None:
    assert_that(response.status == status, f"{label}: expected HTTP {status}, got {response.status}")
    if code is not None:
        assert_that(
            isinstance(response.body, dict) and str(response.body.get("code", "")) == code,
            f"{label}: expected error code {code}",
        )


def encode_multipart(fields: dict[str, str], filename: str, file_bytes: bytes,
                     content_type: str = "text/markdown") -> tuple[bytes, str]:
    boundary = "ragstudio-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
             f"{value}\r\n").encode("utf-8")
        )
    # filename is generated locally; reject header injection.
    safe_name = Path(filename).name.replace("\r", "").replace("\n", "").replace('"', "")
    chunks.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{safe_name}\"\r\nContent-Type: {content_type}\r\n\r\n").encode()
        + file_bytes + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CheckError("HTTP redirects are disabled for local acceptance checks")


class HttpClient:
    """Small cookie-preserving JSON HTTP client; does not require requests/httpx."""

    def __init__(self, base_url: str, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        target = urlsplit(self.base_url)
        assert_that(
            target.scheme in ("http", "https") and target.hostname in ("127.0.0.1", "localhost", "::1")
            and target.username is None and target.password is None
            and not (target.path or target.query or target.fragment),
            "Only local HTTP or HTTPS service origins are accepted",
        )
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect(), HTTPCookieProcessor(CookieJar()))
        self.csrf = ""

    def request(self, method: str, path: str, body: Any = None,
                headers: dict[str, str] | None = None,
                multipart: tuple[bytes, str] | None = None) -> ApiResponse:
        assert_that(path.startswith("/") and not path.startswith("//"), "API path must be relative")
        if body is not None and multipart is not None:
            raise ValueError("Choose JSON or multipart body")
        http_headers = dict(headers or {})
        payload = None
        if multipart is not None:
            payload, content_type = multipart
            http_headers["Content-Type"] = content_type
        elif body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            http_headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=payload, headers=http_headers, method=method)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        except URLError as exc:
            raise CheckError(f"{method} {path}: cannot reach local service ({exc.reason})") from exc
        with response:
            raw = response.read(1024 * 1024).decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw.strip() else None
            except ValueError:
                parsed = None
            return ApiResponse(response.status, parsed, raw)

    def ok(self, method: str, path: str, body: Any = None,
           headers: dict[str, str] | None = None) -> Any:
        response = self.request(method, path, body=body, headers=headers)
        assert_that(200 <= response.status < 300, f"{method} {path} failed HTTP {response.status}; error code "
                    f"{response.body.get('code', 'UNKNOWN') if isinstance(response.body, dict) else 'UNKNOWN'}")
        return response.body

    def login(self, environment: str, username: str, password: str) -> None:
        csrf = self.ok("GET", "/api/v1/auth/csrf")
        token = (csrf or {}).get("csrfToken")
        assert_that(token, "CSRF endpoint returned no token")
        response = self.request(
            "POST", "/api/v1/auth/login",
            # All three isolated environments seed the same demo tenant.
            body={"tenantCode": "demo", "loginName": username, "password": password},
            headers={"X-CSRF-Token": str(token)},
        )
        assert_http(response, 200, label=f"login {username}")
        assert_that((response.body or {}).get("authenticated") is True,
                    f"login {username} did not authenticate")
        fresh = self.ok("GET", "/api/v1/auth/csrf")
        self.csrf = str(fresh["csrfToken"])

    def headers(self, *, idempotent: bool = True) -> dict[str, str]:
        assert_that(self.csrf, "Log in before issuing a mutation")
        result = {"X-CSRF-Token": self.csrf}
        if idempotent:
            result["Idempotency-Key"] = str(uuid.uuid4())
        return result


def local_quote(text: str) -> str:
    return quote(text, safe="")


def report_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log_error(exc: Exception) -> int:
    # Never stringify arbitrary HTTP bodies or environment values.
    print(f"Error: {exc}", file=sys.stderr)
    return 1
