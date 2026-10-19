import re
from typing import List


def normalize_terms(text: str) -> str:
    """
    Normalizes text to optimize BM25 keyword matching for:
    - snake_case (e.g. active_build_id -> active_build_id active build id)
    - camelCase (e.g. expectedVersion -> expectedVersion expected version)
    - paths (e.g. backend/src/main.py -> backend src main py)
    - error codes (e.g. INVALID_MODEL_RESPONSE -> INVALID_MODEL_RESPONSE invalid model response)
    """
    if not text:
        return ""

    tokens: List[str] = []

    # 第 1 步：识别代码标识符与路径；这些 token 对技术文档 BM25 召回非常关键，不能只按自然语言分词。
    # 匹配文件/模块路径，并保留完整路径与可拆分片段，兼顾精确搜索和局部命中。
    paths = re.findall(r"[\w.-]+(?:/[\w.-]+)+", text)
    for p in paths:
        tokens.append(p)
        tokens.extend(re.split(r"[/\\.-]", p))

    # 匹配 snake_case 与错误码；同时保留原词和拆分词，支持用户输入任一种形式。
    snake_words = re.findall(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+", text)
    for sw in snake_words:
        tokens.append(sw)
        tokens.extend(sw.split("_"))

    # 匹配 camelCase，并补充拆词结果，让 expectedVersion 与 expected version 都能召回同一内容。
    camel_words = re.findall(r"[a-z]+(?:[A-Z][a-z0-9]+)+", text)
    for cw in camel_words:
        tokens.append(cw)
        tokens.extend(re.findall(r"[a-z]+|[A-Z][a-z0-9]*", cw))

    # 清理空 token 与无检索价值的单字符非 ASCII token，减少 BM25 噪声但保留代码标识符。
    cleaned = []
    seen = set()
    for t in tokens:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            cleaned.append(t_clean)

    return f"{text} {' '.join(cleaned)}".strip()
