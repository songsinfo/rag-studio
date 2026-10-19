"""统一文档解析入口。

当前项目的设计原则是：所有支持的文件最终都走 Docling + HybridChunker。
本文件负责“输入格式适配 + 调用 Docling + 校验返回值”，不再自行实现第二套切分逻辑。
这样可以避免 PDF/DOCX/Markdown/TXT 各自走不同解析器导致行为漂移。
"""
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

from app.config import settings
from app.errors import ApiException
from app.knowledge.chunker import adapt_docling_hybrid_chunks


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


def _prepare_docling_source(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """把少数文本型输入规范化成 Docling 更稳定处理的格式。"""
    lower = filename.lower()
    # TXT 只做编码规范化，不在本地切块；真正解析/切块仍交给 Docling。
    if lower.endswith((".txt", ".text")):
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ApiException(422, "ENCODING_INVALID", "文本文件必须为 UTF-8") from exc
        return text.encode("utf-8"), filename

    # FAQ JSONL 先转换成 Markdown 语义结构，再交给统一的 Docling 流程。
    # 这样 FAQ 不需要维护独立 chunker，同时保留“问题/答案”的结构提示。
    if lower.endswith(".jsonl"):
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ApiException(422, "ENCODING_INVALID", "FAQ JSONL 必须为 UTF-8") from exc
        sections = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            # 空行允许存在；真正的数据行必须是包含 question/answer 的 JSON 对象。
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ApiException(422, "FAQ_INVALID", f"FAQ 第 {line_no} 行不是合法 JSON") from exc
            if not isinstance(row, dict):
                raise ApiException(422, "FAQ_INVALID", f"FAQ 第 {line_no} 行必须是对象")
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                raise ApiException(422, "FAQ_INVALID", f"FAQ 第 {line_no} 行缺少 question/answer")
            sections.append(
                f"# FAQ {line_no}\n\n**问题：** {question}\n\n**答案：** {answer}"
            )
        if not sections:
            raise ApiException(422, "EMPTY_DOCUMENT", "FAQ JSONL 没有有效内容")
        return ("\n\n".join(sections) + "\n").encode("utf-8"), f"{Path(filename).stem}.md"

    return file_bytes, filename


def _docling_hybrid_chunks(
    file_bytes: bytes,
    filename: str,
    *,
    max_tokens: int,
    tokenizer: str,
    merge_peers: bool,
    vision: bool,
) -> List[Dict[str, Any]]:
    """调用 Docling HybridChunker 完成解析与 child chunk 切分。

    这里故意把 OCR、表格、图片理解和 chunk 参数一次性传给 Docling，
    避免“解析阶段一套参数、切块阶段另一套参数”导致不可复现。
    """
    base_url = (os.getenv("DOCLING_URL") or settings.DOCLING_URL).rstrip("/")
    api_key = os.getenv("DOCLING_API_KEY") or settings.DOCLING_API_KEY
    # Docling 是当前统一解析依赖；未配置时直接失败，不能静默退回另一套本地解析逻辑。
    if not base_url:
        raise ApiException(503, "DOCLING_UNAVAILABLE", "未配置 Docling 服务地址")

    headers = {"X-Api-Key": api_key} if api_key else {}
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    data = {
        # 转换阶段参数：尽量保留文档结构，遇到转换错误直接失败。
        # Docling 转换参数集中在这里维护；不同输入格式只会应用与自身相关的选项，避免调用层重复判断文件类型。
        "convert_do_ocr": "true",
        "convert_do_table_structure": "true",
        "convert_table_mode": "accurate",
        "convert_abort_on_error": "true",
        "convert_do_pdf_heading_hierarchy": "true",
        "convert_do_picture_classification": _bool_form(vision),
        "convert_do_picture_description": _bool_form(vision),
        # 切块阶段参数：child chunk 的 token 边界由 Docling HybridChunker 负责。
        # HybridChunker 的切分参数单独配置；解析负责结构化文档，chunker 再据此生成最终可检索块。
        "chunking_use_markdown_tables": "true",
        "chunking_include_raw_text": "true",
        "chunking_max_tokens": str(max_tokens),
        "chunking_merge_peers": _bool_form(merge_peers),
        "include_converted_doc": "false",
        "target_type": "inbody",
    }
    if tokenizer:
        data["chunking_tokenizer"] = tokenizer

    try:
        # 连接超时短、整体处理超时长：文档解析可能耗时，但服务不可达应尽快失败。
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
            response = client.post(
                f"{base_url}/v1/chunk/hybrid/file",
                headers=headers,
                files={"files": (Path(filename).name, file_bytes, media_type)},
                data=data,
            )
    except Exception as exc:
        raise ApiException(503, "DOCLING_UNAVAILABLE", f"Docling 连接失败: {exc}") from exc

    # 非 200 不做“尽量解析”；解析产物是后续索引的事实基础，失败必须显式暴露。
    if response.status_code != 200:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload.get("error") or "")
        except Exception:
            detail = str(getattr(response, "text", "") or "")
        suffix = f": {detail[:300]}" if detail else ""
        raise ApiException(
            502,
            "DOCLING_PARSE_FAILED",
            f"Docling HybridChunker 处理失败，状态码 {response.status_code}{suffix}",
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise ApiException(502, "DOCLING_INVALID_RESPONSE", "Docling 未返回合法 JSON") from exc

    # 下游只接受 Docling 已完成的 Hybrid chunks；空列表不能当作“成功但无内容”。
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ApiException(
            502,
            "DOCLING_INVALID_RESPONSE",
            "Docling HybridChunker 未返回有效 chunks",
        )

    # warning 不阻断入库，但挂到第一个 chunk 的 metadata 中，保证后续可追溯。
    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    if isinstance(warnings, list) and warnings:
        first = dict(chunks[0])
        metadata = dict(first.get("metadata") or {})
        metadata["doclingWarnings"] = warnings[:100]
        first["metadata"] = metadata
        chunks = [first, *chunks[1:]]
    return chunks


def parse_and_chunk_document(
    file_bytes: bytes,
    filename: str,
    build_config: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """统一入库边界：所有文件均由 Docling 解析，并由 HybridChunker 生成 child chunks。"""
    cfg = build_config or {}
    max_tokens = int(
        cfg.get("chunkMaxTokens")
        or os.getenv("DOCLING_CHUNK_MAX_TOKENS")
        or settings.DOCLING_CHUNK_MAX_TOKENS
    )
    # 防止异常配置把 chunk 调得过小/过大；限制范围同时也是资源保护。
    max_tokens = max(64, min(4096, max_tokens))
    parent_max_tokens = int(
        cfg.get("parentMaxTokens")
        or os.getenv("DOCLING_PARENT_MAX_TOKENS")
        or settings.DOCLING_PARENT_MAX_TOKENS
    )
    # parent 必须至少能容纳一个 child，并设置更高上限避免父上下文无限膨胀。
    parent_max_tokens = max(max_tokens, min(8192, parent_max_tokens))
    tokenizer = str(
        cfg.get("chunkTokenizer")
        or os.getenv("DOCLING_CHUNK_TOKENIZER")
        or settings.DOCLING_CHUNK_TOKENIZER
        or ""
    ).strip()
    merge_peers = str(
        cfg.get("mergePeers")
        if "mergePeers" in cfg
        else os.getenv("DOCLING_CHUNK_MERGE_PEERS", str(settings.DOCLING_CHUNK_MERGE_PEERS))
    ).lower() in {"1", "true", "yes", "on"}

    # 这里只做格式适配；从这里开始所有格式进入同一 Docling 路径。
    prepared_bytes, prepared_filename = _prepare_docling_source(file_bytes, filename)
    docling_chunks = _docling_hybrid_chunks(
        prepared_bytes,
        prepared_filename,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        merge_peers=merge_peers,
        vision=bool(cfg.get("vision", False)),
    )
    # Docling 决定 child 如何切；本地 adapter 只负责组装 SourceUnit/Parent/Span 追溯关系。
    return adapt_docling_hybrid_chunks(
        docling_chunks,
        parent_max_tokens=parent_max_tokens,
    )


def parse_document(
    file_bytes: bytes,
    filename: str,
    build_config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper. New ingestion should use parse_and_chunk_document."""
    units, _, _ = parse_and_chunk_document(file_bytes, filename, build_config)
    return units
