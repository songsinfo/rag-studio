import re
from typing import Dict, Tuple


_INDIRECT_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|"
    r"system\s+prompt|developer\s+message|"
    r"reveal\s+.*(secret|password|token)|"
    r"do\s+not\s+follow\s+the\s+user)",
    re.IGNORECASE,
)


def guard_retrieved_content(text: str) -> Tuple[str, Dict[str, object]]:
    raw = str(text or "")
    suspicious = bool(_INDIRECT_INJECTION.search(raw))
    # 检索内容始终放在明确的数据边界中，并按“不可信数据”处理；即使包含疑似 prompt injection，
    # 这些文本仍可作为证据引用，但绝不能伪装成 system/tool 指令或改变 Agent 控制流。
    wrapped = (
        '<retrieved_document trust="untrusted-data">\n'
        + raw
        + "\n</retrieved_document>"
    )
    return wrapped, {"indirectPromptInjection": suspicious}
