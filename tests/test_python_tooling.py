"""Offline regression tests of migrated PowerShell semantics, no network or credentials."""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.request import HTTPRedirectHandler, Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acceptance
import dev
import evaluation
import integration
import langfuse_local
import tooling


def sample(case_key="x"):
    return {
        "caseKey": case_key, "groupKey": case_key, "split": "TEST", "category": "FACT",
        "question": {"text": "Test?"}, "persona": {"loginName": "admin"},
        "expected": {"status": "ANSWER", "evidenceGroups": [{
            "fact": "Fact", "alternatives": [{
                "documentId": str(uuid.uuid4()), "versionId": str(uuid.uuid4()),
                "quote": "Fact", "relevanceGrade": 3
            }]
        }]}, "annotationStatus": "AUTO_REVIEWED"
    }


class PythonToolingTests(unittest.TestCase):
    def test_http_assertions_reject_false_positives(self):
        integration.self_test()
        tooling.assert_http(tooling.ApiResponse(409, {"code": "EXPECTED"}, ""), 409, "EXPECTED")
        with self.assertRaises(tooling.CheckError):
            tooling.assert_http(tooling.ApiResponse(409, {"code": "WRONG"}, ""), 409, "EXPECTED")

    def test_multipart_escaping(self):
        data, mime = tooling.encode_multipart({"kbId": "a"}, "x.md", b"hello")
        self.assertIn("multipart/form-data; boundary=", mime)
        self.assertIn(b"hello", data)
        self.assertIn(b'name="kbId"', data)

    def test_valid_dataset_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            row = sample()
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = evaluation.validate(path, expected_count=1, require_runtime_bindings=True)
            self.assertEqual(report["cases"], 1)
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(tooling.CheckError):
                evaluation.validate(path)

    def test_bind_template_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = sample()
            row["persona"]["loginName"] = "operator"
            row["expected"]["evidenceGroups"][0]["alternatives"][0]["documentId"] = "fixture.md"
            inp, mapping, out = (root / name for name in ("in.jsonl", "bindings.json", "out.jsonl"))
            inp.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(tooling.CheckError, "template created"):
                evaluation.bind(inp, mapping, out)
            contents = json.loads(mapping.read_text())
            contents["sources"]["fixture.md"] = {"documentId": str(uuid.uuid4()),
                                                      "versionId": str(uuid.uuid4())}
            mapping.write_text(json.dumps(contents))
            evaluation.bind(inp, mapping, out)
            self.assertEqual(evaluation.validate(out, require_runtime_bindings=True)["cases"], 1)

    def test_ablation_contract(self):
        configs = evaluation.configurations(ROOT / "experiments" / "retrieval-ablation.json")
        self.assertEqual([c["id"] for c in configs], ["B0", "B1", "B2", "B3"])

    def test_paid_cost_cap_checked_before_invocation(self):
        for invalid in (0, -1, 5.01):
            with self.assertRaises(tooling.CheckError):
                acceptance.real_acceptance("model", invalid)

    def test_dev_python_executable_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tooling, "ROOT", Path(tmp)):
            self.assertEqual(tooling.python_executable(), sys.executable)

    def test_local_only_http(self):
        for url in ("http://example.invalid:8080", "https://example.invalid",
                    "http://localhost:8080@example.invalid", "http://localhost:8080#fragment"):
            with self.subTest(url=url), self.assertRaises(tooling.CheckError):
                tooling.HttpClient(url)
        for url in ("http://127.0.0.1:8080", "http://localhost:8080", "https://[::1]:8080"):
            tooling.HttpClient(url)

    def test_http_redirect_cannot_forward_local_credentials(self):
        client = tooling.HttpClient("http://127.0.0.1:8080")
        handler = next(h for h in client.opener.handlers if isinstance(h, HTTPRedirectHandler))
        request = Request(client.base_url + "/models", headers={"Authorization": "Bearer synthetic"})
        with self.assertRaises(tooling.CheckError):
            handler.redirect_request(request, None, 302, "Found", {}, "https://example.invalid/models")

    def test_environment_is_not_the_seeded_tenant_code(self):
        client = tooling.HttpClient("http://127.0.0.1:8080")
        with patch.object(client, "ok", side_effect=[{"csrfToken": "first"}, {"csrfToken": "fresh"}]), \
                patch.object(client, "request", return_value=tooling.ApiResponse(200, {"authenticated": True}, "")) as request:
            client.login("evaluation", "admin", "synthetic-password")
        self.assertEqual(request.call_args.kwargs["body"]["tenantCode"], "demo")

    def test_langfuse_compose_uses_private_file_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lf.env"
            path.write_text("LFD_POSTGRES_PASSWORD=synthetic-file-value\n", encoding="utf-8")
            with patch.object(langfuse_local, "LOCAL_FILE", path), \
                    patch.dict(os.environ, {"LFD_POSTGRES_PASSWORD": "synthetic-shell-value"}), \
                    patch.object(langfuse_local.subprocess, "run") as run:
                langfuse_local.run_compose("up", "-d")
            self.assertNotIn("LFD_POSTGRES_PASSWORD", run.call_args.kwargs["env"])

    def test_langfuse_restores_services_even_if_stop_partially_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lf.env"
            path.write_text("", encoding="utf-8")
            with patch.object(langfuse_local, "LOCAL_FILE", path), \
                    patch.object(langfuse_local, "run_compose", side_effect=[
                        subprocess.CalledProcessError(1, ["docker", "compose"]), None]) as compose, \
                    patch.object(langfuse_local.subprocess, "run") as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    langfuse_local.check(outage=True, content=False)
            self.assertEqual([call.args for call in compose.call_args_list], [
                ("stop", "web", "worker"), ("start", "web", "worker")])
            run.assert_not_called()

    def test_langfuse_init_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lf.env"
            with patch.object(langfuse_local, "LOCAL_FILE", path):
                langfuse_local.initialize()
                old = path.read_bytes()
                with self.assertRaises(tooling.CheckError):
                    langfuse_local.initialize()
                self.assertEqual(old, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
