"""把 Docling HybridChunker 的结果适配为项目内部的 Parent-Child + Provenance 结构。

重要边界：
- child 的真正切分由 Docling 完成，本文件不再次把 child 重新切碎；
- 本文件只做相邻 child 的 parent 分组、SourceUnit 组装和字符位置映射；
- 这样既保留 HybridChunker 的语义切分，又支持 read_section、引用回溯和 parent expansion。
"""
import hashlib
import uuid
from typing import Any, Dict, List, Tuple


def _heading_key(item: Dict[str, Any]) -> Tuple[str, ...]:
    """把标题路径规范化成可比较的 tuple，用于判断相邻 chunk 是否属于同一章节。"""
    headings = item.get("headings")
    if not isinstance(headings, list):
        return ()
    return tuple(str(value).strip() for value in headings if str(value).strip())


def _location(item: Dict[str, Any]) -> Dict[str, Any]:
    """保留 Docling 给出的页码、标题、caption、doc item 等来源定位信息。"""
    location: Dict[str, Any] = {
        "doclingChunkIndex": int(item.get("chunk_index") or 0),
        "headings": list(_heading_key(item)),
        "captions": list(item.get("captions") or []),
        "docItems": list(item.get("doc_items") or []),
        "pageNumbers": list(item.get("page_numbers") or []),
    }
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata:
        location["metadata"] = metadata
    return location


def _group_location(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把一个 parent 分组内多个 child 的定位信息合并成 parent/SourceUnit 级 provenance。"""
    headings = list(group[0]["headings"]) if group else []
    # 页码去重排序，保证同一输入得到稳定 metadata，方便比较和测试。
    pages = sorted({
        int(page)
        for item in group
        for page in item["location"].get("pageNumbers", [])
        if isinstance(page, int)
    })
    doc_items = [
        str(value)
        for item in group
        for value in item["location"].get("docItems", [])
    ]
    captions = [
        str(value)
        for item in group
        for value in item["location"].get("captions", [])
    ]
    metadata: Dict[str, Any] = {}
    warnings: List[Any] = []
    # 普通 metadata 首个值优先；Docling warnings 则汇总，避免后续 child 的 warning 丢失。
    for item in group:
        item_metadata = item["location"].get("metadata")
        if not isinstance(item_metadata, dict):
            continue
        for key, value in item_metadata.items():
            if key == "doclingWarnings" and isinstance(value, list):
                warnings.extend(value)
            elif key not in metadata:
                metadata[key] = value
    if warnings:
        metadata["doclingWarnings"] = warnings

    location = {
        "headings": headings,
        "pageNumbers": pages,
        "docItems": doc_items,
        "captions": captions,
        "doclingChunkStart": int(group[0]["chunk_index"]) if group else 0,
        "doclingChunkEnd": int(group[-1]["chunk_index"]) if group else 0,
    }
    if metadata:
        location["metadata"] = metadata
    return location


def adapt_docling_hybrid_chunks(
    items: List[Dict[str, Any]],
    *,
    parent_max_tokens: int = 1200,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把 Docling HybridChunker 输出转换成项目内部追溯结构。

    Docling 已经决定 child 的切分边界和 token 数；这里仅把“标题路径相同且 parent
    token 不超预算”的相邻 child 合并为 parent 和 SourceUnit，不重新切 child。
    """
    if not items:
        return [], [], []

    # 第一步：把 Docling 不同字段形态归一化，后续分组只处理稳定结构。
    normalized: List[Dict[str, Any]] = []
    for position, item in enumerate(items):
        text = str(item.get("text") or "").strip()
        raw_text = str(item.get("raw_text") or text).strip()
        if not text:
            continue
        if not raw_text:
            raw_text = text
        normalized.append(
            {
                "chunk_index": int(
                    item.get("chunk_index")
                    if item.get("chunk_index") is not None
                    else position
                ),
                "text": text,
                "raw_text": raw_text,
                # Docling 正常会返回 num_tokens；缺失时用字符数粗略兜底，仅用于 parent 预算估算。
                "num_tokens": max(
                    1,
                    int(item.get("num_tokens") or max(1, len(text) // 4)),
                ),
                "headings": _heading_key(item),
                "location": _location(item),
            }
        )

    # 第二步：构建 parent 分组。
    # 分组只在“标题变化”或“parent token 预算超限”时断开。
    groups: List[List[Dict[str, Any]]] = []
    group: List[Dict[str, Any]] = []
    group_tokens = 0
    group_heading: Tuple[str, ...] | None = None

    for item in normalized:
        heading = item["headings"]
        would_exceed = bool(group) and group_tokens + item["num_tokens"] > parent_max_tokens
        heading_changed = bool(group) and heading != group_heading
        # 任何一个条件触发都先结束当前 parent，再从当前 child 开始新 parent。
        if would_exceed or heading_changed:
            groups.append(group)
            group = []
            group_tokens = 0
            group_heading = None
        if not group:
            group_heading = heading
        group.append(item)
        group_tokens += item["num_tokens"]
    if group:
        groups.append(group)

    # 第三步：为每个 parent 分组生成 SourceUnit、Parent chunk、Child chunk 和 span。
    units: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    child_ordinal = 0

    for parent_ordinal, group in enumerate(groups, start=1):
        # SourceUnit 表示可回读的原文段；child 的 span 都指向它。
        source_unit_id = str(uuid.uuid4())
        raw_parts = [item["raw_text"] for item in group]
        source_text = "\n\n".join(raw_parts)
        location = _group_location(group)
        units.append(
            {
                "id": source_unit_id,
                "unit_key": f"docling_section_{parent_ordinal}",
                "kind": "TEXT",
                "text_content": source_text,
                "location": location,
                "content_hash": hashlib.sha256(source_text.encode("utf-8")).digest(),
            }
        )

        headings = list(group[0]["headings"])
        heading_prefix = "\n".join(headings).strip()
        parent_body = source_text
        parent_text = (
            f"{heading_prefix}\n{parent_body}".strip()
            if heading_prefix
            else parent_body
        )
        # Parent 只用于扩展上下文，不直接承载 source span；真正检索命中的是 child。
        parent_id = str(uuid.uuid4())
        chunks.append(
            {
                "id": parent_id,
                "parent_id": None,
                "kind": "PARENT",
                "ordinal": parent_ordinal,
                "text_content": parent_text,
                "token_count": sum(item["num_tokens"] for item in group),
                "content_hash": hashlib.sha256(parent_text.encode("utf-8")).digest(),
            }
        )

        cursor = 0
        for item in group:
            child_ordinal += 1
            child_id = str(uuid.uuid4())
            chunks.append(
                {
                    "id": child_id,
                    "parent_id": parent_id,
                    "kind": "CHILD",
                    "ordinal": child_ordinal,
                    "text_content": item["text"],
                    "token_count": item["num_tokens"],
                    "content_hash": hashlib.sha256(item["text"].encode("utf-8")).digest(),
                }
            )
            # span 使用 raw_text 在 source_text 中的字符偏移，保证引用可以回到原始段落。
            start = cursor
            end = start + len(item["raw_text"])
            spans.append(
                {
                    "id": str(uuid.uuid4()),
                    "chunk_id": child_id,
                    "unit_id": source_unit_id,
                    "ordinal": 1,
                    "start_cp": start,
                    "end_cp": end,
                    "location": item["location"],
                }
            )
            # source_text 中 child 之间固定用两个换行拼接，所以游标同步跳过 2 个字符。
            cursor = end + 2  # "\n\n" separator in source_text

    return units, chunks, spans
