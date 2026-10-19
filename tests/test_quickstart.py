"""Offline contract tests for the cross-platform Quick Start (no Docker required)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "quickstart.py"
SPEC = importlib.util.spec_from_file_location("quickstart", MODULE_PATH)
quickstart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quickstart)


class QuickstartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / ".env.example").write_text(
            (MODULE_PATH.parents[1] / ".env.example").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.patch_root = patch.object(quickstart, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def test_init_is_private_idempotent_and_random(self):
        quickstart.initialize("demo")
        path = self.root / ".runtime" / "demo.env"
        first = path.read_bytes()
        self.assertNotIn(b"DB_PASSWORD=\n", first)
        self.assertNotIn(b"RAG_CSRF_SECRET=\n", first)
        quickstart.initialize("demo")
        self.assertEqual(path.read_bytes(), first)
        quickstart.initialize("evaluation")
        other = quickstart.read_env(self.root / ".runtime" / "evaluation.env")
        first_values = quickstart.read_env(path)
        self.assertNotEqual(first_values["DB_PASSWORD"], other["DB_PASSWORD"])
        self.assertEqual(first_values["RAG_WEB_PORT"], "5173")
        self.assertEqual(other["RAG_WEB_PORT"], "5174")

    def test_missing_models_fail_before_docker(self):
        quickstart.initialize("demo")
        with self.assertRaisesRegex(ValueError, "MODEL_BASE_URL"):
            quickstart.check_configuration("demo")

    def test_compose_includes_all_files_and_profiles(self):
        quickstart.initialize("demo")
        values = quickstart.read_env(quickstart.env_path("demo"))
        cmd = quickstart.compose_command("demo", values)
        self.assertIn("app.compose.yml", " ".join(cmd))
        self.assertIn("documents", cmd)
        self.assertNotIn("artifacts-s3", cmd)
        values["RAG_ARTIFACT_STORE"] = "s3"
        self.assertIn("artifacts-s3", quickstart.compose_command("demo", values))

    def test_no_overwrite_existing_config(self):
        path = self.root / ".runtime" / "demo.env"
        path.parent.mkdir(parents=True)
        path.write_text("RAG_SEED_PASSWORD=user-secret\n", encoding="utf-8")
        quickstart.initialize("demo")
        self.assertEqual(path.read_text(encoding="utf-8"), "RAG_SEED_PASSWORD=user-secret\n")

    def test_docker_uses_selected_env_file_instead_of_ambient_credentials(self):
        values = {"RAG_ENVIRONMENT": "evaluation", "MODEL_API_KEY": "synthetic-file-key"}
        with patch.dict(os.environ, {"RAG_ENVIRONMENT": "demo", "MODEL_API_KEY": "synthetic-shell-key"}), \
                patch.object(quickstart.subprocess, "run") as run:
            quickstart.run_docker(["docker", "compose", "config"], values)
        child_env = run.call_args.kwargs["env"]
        self.assertNotIn("RAG_ENVIRONMENT", child_env)
        self.assertNotIn("MODEL_API_KEY", child_env)
        self.assertEqual(child_env.get("PATH"), os.environ.get("PATH"))


if __name__ == "__main__":
    unittest.main()
