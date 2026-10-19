"""从真正编译的 LangGraph StateGraph 导出普通 RAG 检索图。

用法（仓库的 backend 目录）：
    python scripts/export_rag_retrieval_graph.py --print
    python scripts/export_rag_retrieval_graph.py --write
    python scripts/export_rag_retrieval_graph.py --check

不执行检索，也不会调用模型、Milvus 或数据库查询：只加载图定义并编译。
--check 可放在 CI 中防止文档图示与实际节点/边发生漂移。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = BACKEND_DIR.parent
DOC_PATH = REPOSITORY_DIR / "docs" / "rag-langgraph-retrieval.md"
BEGIN = "<!-- compiled-rag-retrieval-graph:begin -->"
END = "<!-- compiled-rag-retrieval-graph:end -->"


def compiled_mermaid() -> str:
    # scripts/ 不是 app 包的一部分；从 backend 根路径导入正式业务代码。
    sys.path.insert(0, str(BACKEND_DIR))
    from app.retrieval.rag_graph import RagRetrievalGraph

    compiled = RagRetrievalGraph._get_graph()
    return compiled.get_graph().draw_mermaid().strip()


def figure(mermaid: str) -> str:
    return f"{BEGIN}\n\x60\x60\x60mermaid\n{mermaid}\n\x60\x60\x60\n{END}"


def main() -> int:
    parser = argparse.ArgumentParser(description="从编译后的 RagRetrievalGraph 生成 Mermaid")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true", help="输出编译后的 Mermaid 源码")
    mode.add_argument("--write", action="store_true", help="更新 docs 中的自动生成区块")
    mode.add_argument("--check", action="store_true", help="验证 docs 中的图与编译结果完全一致")
    args = parser.parse_args()
    block = figure(compiled_mermaid())
    if args.print:
        print(block)
        return 0

    current = DOC_PATH.read_text(encoding="utf-8")
    if current.count(BEGIN) != 1 or current.count(END) != 1:
        raise RuntimeError(f"{DOC_PATH}: 必须恰好有一个完整的自动生成图标记区间")

    start = current.index(BEGIN)
    stop = current.index(END, start) + len(END)
    existing = current[start:stop]
    if args.check:
        if existing != block:
            print("编译图与文档不一致。请在 backend 目录运行："
                  " python scripts/export_rag_retrieval_graph.py --write", file=sys.stderr)
            return 1
        print("编译后的 LangGraph 图与 docs/rag-langgraph-retrieval.md 一致")
        return 0

    DOC_PATH.write_text(current[:start] + block + current[stop:], encoding="utf-8")
    print(f"已更新 {DOC_PATH.relative_to(REPOSITORY_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
