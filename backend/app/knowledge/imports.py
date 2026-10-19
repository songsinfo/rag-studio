import base64
import hashlib
import mimetypes
import os
import re
import threading
import time
import urllib.parse
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from app.errors import ApiException


_PREVIEWS: Dict[str, Dict[str, Any]] = {}
_PREVIEW_LOCK = threading.Lock()
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def _import_root() -> Path:
    configured = os.getenv("RAG_IMPORT_ROOT", "").strip()
    if not configured:
        raise ApiException(503, "IMPORT_UNAVAILABLE", "未配置 RAG_IMPORT_ROOT")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise ApiException(503, "IMPORT_UNAVAILABLE", "RAG_IMPORT_ROOT 不存在或不是目录")
    return root


def _safe_import_path(relative_path: str) -> Path:
    value = str(relative_path or "").strip().replace("\\", "/")
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ApiException(400, "INVALID_INPUT", "relativePath 无效")
    root = _import_root()
    target = (root / value).resolve()
    if not target.is_relative_to(root):
        raise ApiException(400, "INVALID_INPUT", "relativePath 越界")
    return target


def create_import_preview(tenant_id: str, user_id: str, relative_path: str) -> Dict[str, Any]:
    target = _safe_import_path(relative_path)
    candidates = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    if len(candidates) > 1000:
        raise ApiException(413, "INPUT_TOO_LARGE", "导入目录文件数量超过1000")

    root = _import_root()
    files = []
    for path in candidates:
        if not path.is_relative_to(root):
            continue
        size = path.stat().st_size
        if size > 25 * 1024 * 1024:
            continue
        data = path.read_bytes()
        files.append(
            {
                "relativePath": path.relative_to(root).as_posix(),
                "sizeBytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    now = time.time()
    with _PREVIEW_LOCK:
        expired = [key for key, value in _PREVIEWS.items() if value["expires"] < now]
        for key in expired:
            _PREVIEWS.pop(key, None)
        if len(_PREVIEWS) >= 100:
            raise ApiException(429, "RATE_LIMITED", "导入预览过多")
        estimate_id = str(uuid.uuid4())
        _PREVIEWS[estimate_id] = {
            "tenantId": tenant_id,
            "userId": user_id,
            "expires": now + 600,
            "files": {item["relativePath"]: item for item in files},
        }

    return {
        "estimateId": estimate_id,
        "files": files,
        "estimatedModelCost": 0,
        "mode": "DOCLING",
    }


def consume_import_preview(
    tenant_id: str,
    user_id: str,
    estimate_id: str,
    selected: List[str],
) -> List[Dict[str, Any]]:
    if not selected or len(selected) > 100:
        raise ApiException(400, "INVALID_INPUT", "files 必须包含1到100个文件")

    now = time.time()
    with _PREVIEW_LOCK:
        preview = _PREVIEWS.get(estimate_id)
    if (
        not preview
        or preview["expires"] < now
        or preview["tenantId"] != tenant_id
        or preview["userId"] != user_id
    ):
        raise ApiException(409, "ESTIMATE_EXPIRED", "导入预览已失效")

    output = []
    for relative_path in selected:
        entry = preview["files"].get(str(relative_path))
        if not entry:
            raise ApiException(400, "INVALID_INPUT", "文件不在预览结果中")
        path = _safe_import_path(str(relative_path))
        if not path.is_file():
            raise ApiException(409, "SOURCE_CHANGED", "导入文件已不存在")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ApiException(409, "SOURCE_CHANGED", "导入文件已变化")
        output.append(
            {
                "relativePath": str(relative_path),
                "filename": path.name,
                "bytes": data,
            }
        )
    return output


def _safe_zip_name(name: str) -> str:
    decoded = urllib.parse.unquote(str(name or "")).replace("\\", "/")
    path = Path(decoded)
    if not decoded or decoded.startswith("/") or ".." in path.parts:
        raise ApiException(400, "INVALID_ARCHIVE", "Wiki 包包含非法路径")
    return path.as_posix()


def split_wiki_package(data: bytes) -> List[Dict[str, Any]]:
    if len(data) > 25 * 1024 * 1024:
        raise ApiException(413, "INPUT_TOO_LARGE", "Wiki包最多25MiB")

    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except Exception as exc:
        raise ApiException(400, "INVALID_ARCHIVE", "Wiki 包不是合法 ZIP") from exc

    infos = archive.infolist()
    if len(infos) > 1000:
        raise ApiException(413, "INPUT_TOO_LARGE", "Wiki 包文件数量超过1000")

    entries: Dict[str, bytes] = {}
    total_uncompressed = 0
    for info in infos:
        if info.is_dir():
            continue
        name = _safe_zip_name(info.filename)
        if info.file_size > 25 * 1024 * 1024:
            raise ApiException(413, "INPUT_TOO_LARGE", "Wiki 包内单文件过大")
        total_uncompressed += info.file_size
        if total_uncompressed > 100 * 1024 * 1024:
            raise ApiException(413, "INPUT_TOO_LARGE", "Wiki 包解压后体积过大")
        if info.compress_size > 0 and info.file_size / info.compress_size > 100:
            raise ApiException(413, "INPUT_TOO_LARGE", "Wiki 包压缩比异常")
        entries[name] = archive.read(info)

    pages = []
    for name, raw in sorted(entries.items()):
        if not name.lower().endswith(".md"):
            continue
        try:
            markdown = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApiException(422, "ENCODING_INVALID", f"Wiki 页面必须为 UTF-8: {name}") from exc

        page_dir = Path(name).parent

        def replace_image(match: re.Match[str]) -> str:
            alt, raw_ref = match.group(1), urllib.parse.unquote(match.group(2))
            lowered = raw_ref.lower()
            if lowered.startswith(("http://", "https://", "//", "data:", "file:")):
                raise ApiException(400, "INVALID_ARCHIVE", "Wiki 页面禁止远程或内联外部图片")
            ref = (page_dir / raw_ref).as_posix()
            safe_ref = _safe_zip_name(ref)
            blob = entries.get(safe_ref)
            if blob is None:
                raise ApiException(400, "INVALID_ARCHIVE", f"Wiki 图片不存在: {raw_ref}")
            if len(blob) > 10 * 1024 * 1024:
                raise ApiException(413, "IMAGE_TOO_LARGE", "Wiki 图片超过10MiB")
            media = mimetypes.guess_type(safe_ref)[0] or "application/octet-stream"
            encoded = base64.b64encode(blob).decode("ascii")
            return f"![{alt}](data:{media};base64,{encoded})"

        markdown = _IMAGE_RE.sub(replace_image, markdown)
        pages.append(
            {
                "filename": Path(name).name,
                "bytes": markdown.encode("utf-8"),
            }
        )

    if not pages:
        raise ApiException(422, "EMPTY_DOCUMENT", "Wiki 包中没有 Markdown 页面")
    if len(pages) > 100:
        raise ApiException(413, "INPUT_TOO_LARGE", "Wiki 页面最多100个")
    return pages
