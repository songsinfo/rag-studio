import hashlib
import os
from pathlib import Path
from typing import Tuple

from app.errors import ApiException


def get_artifact_root() -> Path:
    root_str = os.getenv("RAG_ARTIFACT_ROOT", "")
    if root_str:
        root = Path(root_str).resolve()
    else:
        rag_root = Path(__file__).resolve().parent.parent.parent.parent
        root = rag_root / ".runtime" / os.getenv("RAG_ENVIRONMENT", "demo") / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mode() -> str:
    return os.getenv("RAG_ARTIFACT_STORE", "local").strip().lower()


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise ApiException(503, "ARTIFACT_STORE_UNAVAILABLE", "缺少 boto3，无法使用 S3/MinIO") from exc
    kwargs = {}
    if os.getenv("S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.getenv("S3_ENDPOINT_URL")
    if os.getenv("S3_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = os.getenv("S3_ACCESS_KEY_ID")
    if os.getenv("S3_SECRET_ACCESS_KEY"):
        kwargs["aws_secret_access_key"] = os.getenv("S3_SECRET_ACCESS_KEY")
    kwargs["region_name"] = os.getenv("S3_REGION", "us-east-1")
    return boto3.client("s3", **kwargs)


def _bucket() -> str:
    return os.getenv("S3_BUCKET", "rag-artifacts").strip()


def _ensure_bucket(client, bucket: str):
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=bucket)
        except Exception as exc:
            raise ApiException(503, "ARTIFACT_STORE_UNAVAILABLE", f"对象存储 bucket 不可用: {exc}")


def save_artifact(tenant_id: str, filename: str, data: bytes,
                  document_id: str | None = None, artifact_id: str | None = None) -> Tuple[str, bytes, int]:
    sha = hashlib.sha256(data)
    sha_digest = sha.digest()
    sha_hex = sha.hexdigest()
    # storage_key 是 ASCII 列；原始文件名由 document_version 单独保留。
    safe_name = hashlib.sha256(Path(filename).name.encode("utf-8")).hexdigest()
    storage_key = f"{tenant_id}/{document_id}/{artifact_id}/{safe_name}" if document_id and artifact_id else f"{tenant_id}/{sha_hex}/{safe_name}"
    if _mode() == "s3":
        client, bucket = _s3_client(), _bucket()
        _ensure_bucket(client, bucket)
        try:
            client.put_object(Bucket=bucket, Key=storage_key, Body=data,
                              Metadata={"sha256": sha_hex, "tenant-id": tenant_id})
        except Exception as exc:
            raise ApiException(503, "ARTIFACT_STORE_UNAVAILABLE", f"对象存储写入失败: {exc}")
        return storage_key, sha_digest, len(data)

    root = get_artifact_root()
    dest = root / tenant_id / document_id / artifact_id / safe_name if document_id and artifact_id else root / tenant_id / sha_hex[:2] / sha_hex / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return storage_key, sha_digest, len(data)


def read_artifact(storage_key: str) -> bytes:
    if _mode() == "s3":
        try:
            return _s3_client().get_object(Bucket=_bucket(), Key=storage_key)["Body"].read()
        except Exception as exc:
            raise ApiException(404, "ARTIFACT_NOT_FOUND", f"构件不存在: {exc}")
    root = get_artifact_root()
    target = (root / storage_key).resolve()
    if target.is_file() and target.is_relative_to(root.resolve()):
        return target.read_bytes()
    parts = storage_key.split("/")
    if len(parts) == 3:
        tenant_id, sha_hex, filename = parts
        legacy = (root / tenant_id / sha_hex[:2] / sha_hex / filename).resolve()
        if legacy.is_file() and legacy.is_relative_to(root.resolve()):
            return legacy.read_bytes()
    raise ApiException(404, "ARTIFACT_NOT_FOUND", "构件不存在")


def delete_artifact(storage_key: str):
    if _mode() == "s3":
        try:
            _s3_client().delete_object(Bucket=_bucket(), Key=storage_key)
        except Exception as exc:
            # purge 只有在私有对象存储中的实体文件真正删除后才算完成，不能只删除数据库记录。
            # 存储层删除失败必须继续向上抛出，让 durable job 保持 FAILED/可重试，避免留下无法追踪的敏感文件。
            raise ApiException(503, "ARTIFACT_DELETE_FAILED", f"对象存储删除失败: {exc}") from exc
        return

    root = get_artifact_root()
    target = (root / storage_key).resolve()
    if target.is_file() and target.is_relative_to(root.resolve()):
        try:
            target.unlink()
        except OSError as exc:
            raise ApiException(503, "ARTIFACT_DELETE_FAILED", f"本地构件删除失败: {exc}") from exc
        return

    # 兼容旧版内容寻址路径：历史对象实际可能存放在 tenant/sha[:2]/sha/filename，
    # 但数据库里曾只记录 tenant/sha/filename；清理时同时兼容旧路径，避免升级后产生孤儿对象。
    parts = storage_key.split("/")
    if len(parts) == 3:
        tenant_id, sha_hex, filename = parts
        legacy = (root / tenant_id / sha_hex[:2] / sha_hex / filename).resolve()
        if legacy.is_file() and legacy.is_relative_to(root.resolve()):
            try:
                legacy.unlink()
            except OSError as exc:
                raise ApiException(503, "ARTIFACT_DELETE_FAILED", f"旧格式本地构件删除失败: {exc}") from exc
