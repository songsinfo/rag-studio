import hashlib
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from app.knowledge import artifacts


@pytest.mark.parametrize("filename", ["中文年报.pdf", "报告📄.md", "report.txt"])
@pytest.mark.parametrize("identified", [True, False])
def test_local_artifact_key_is_ascii_and_round_trips(tmp_path, monkeypatch, filename, identified):
    monkeypatch.setenv("RAG_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_ARTIFACT_STORE", "local")
    ids = {"document_id": "document-1", "artifact_id": "artifact-1"} if identified else {}
    data = b"Upload regression test"

    key, digest, size = artifacts.save_artifact("tenant-1", filename, data, **ids)

    assert key.isascii()
    assert len(key) <= 512
    assert digest == hashlib.sha256(data).digest()
    assert size == len(data)
    assert artifacts.read_artifact(key) == data
    artifacts.delete_artifact(key)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_s3_artifact_key_is_ascii_and_round_trips(monkeypatch):
    monkeypatch.setenv("RAG_ARTIFACT_STORE", "s3")
    monkeypatch.setenv("S3_BUCKET", "test-artifacts")
    client = MagicMock()
    monkeypatch.setattr(artifacts, "_s3_client", lambda: client)
    data = b"Upload regression test"

    key, _, _ = artifacts.save_artifact("tenant-1", "中文年报.pdf", data,
                                        document_id="document-1", artifact_id="artifact-1")

    assert key.isascii()
    assert len(key) <= 512
    assert client.put_object.call_args.kwargs["Key"] == key
    assert client.put_object.call_args.kwargs["Body"] == data
    client.get_object.return_value = {"Body": BytesIO(data)}
    assert artifacts.read_artifact(key) == data
    artifacts.delete_artifact(key)
    client.delete_object.assert_called_once_with(Bucket="test-artifacts", Key=key)
