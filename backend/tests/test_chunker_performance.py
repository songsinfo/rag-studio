from app.knowledge.chunker import adapt_docling_hybrid_chunks


def test_docling_hybrid_chunks_are_not_resplit_and_keep_provenance():
    items = [
        {
            "filename": "guide.md",
            "chunk_index": 0,
            "text": "Redis\n第一段",
            "raw_text": "第一段",
            "num_tokens": 120,
            "headings": ["Redis"],
            "captions": [],
            "doc_items": ["#/texts/0"],
            "page_numbers": [1],
            "metadata": {},
        },
        {
            "filename": "guide.md",
            "chunk_index": 1,
            "text": "Redis\n第二段",
            "raw_text": "第二段",
            "num_tokens": 130,
            "headings": ["Redis"],
            "captions": [],
            "doc_items": ["#/texts/1"],
            "page_numbers": [1],
            "metadata": {},
        },
        {
            "filename": "guide.md",
            "chunk_index": 2,
            "text": "Cluster\n第三段",
            "raw_text": "第三段",
            "num_tokens": 100,
            "headings": ["Cluster"],
            "captions": [],
            "doc_items": ["#/texts/2"],
            "page_numbers": [2],
            "metadata": {},
        },
    ]

    units, chunks, spans = adapt_docling_hybrid_chunks(
        items,
        parent_max_tokens=400,
    )
    children = [c for c in chunks if c["kind"] == "CHILD"]
    parents = [c for c in chunks if c["kind"] == "PARENT"]

    # child 边界由 HybridChunker 决定；适配层必须 1:1 保留，不能再次切分导致 provenance 偏移。
    assert [c["text_content"] for c in children] == [item["text"] for item in items]
    assert [c["token_count"] for c in children] == [120, 130, 100]

    # Parent/SourceUnit 只是上下文聚合层，并按 heading path 分组；它不能改变原始 child 的语义边界。
    assert len(parents) == 2
    assert len(units) == 2
    assert units[0]["text_content"] == "第一段\n\n第二段"
    assert units[1]["location"]["pageNumbers"] == [2]

    # 每个 child span 都必须能回指到分组后的 SourceUnit，保证最终引用可追溯。
    assert len(spans) == 3
    assert spans[0]["unit_id"] == spans[1]["unit_id"]
    assert spans[0]["start_cp"] == 0
    assert spans[0]["end_cp"] == len("第一段")
    assert spans[1]["start_cp"] == len("第一段\n\n")
