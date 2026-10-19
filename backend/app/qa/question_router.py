import re
from enum import Enum
from typing import NamedTuple


class Route(str, Enum):
    SMALL_TALK = "SMALL_TALK"
    KNOWLEDGE_QA = "KNOWLEDGE_QA"
    UNSUPPORTED = "UNSUPPORTED"


class Decision(NamedTuple):
    route: Route
    reason: str


SMALL_TALK_PATTERN = re.compile(
    r"(?iu)^\s*(?:你好|您好|hi|hello|谢谢|感谢|再见)[！!。.?？,，\s]*$"
)
UNSUPPORTED_PATTERN = re.compile(
    r"(?iu)^\s*(?:请)?(?:讲个笑话|写一首诗|查询?今天的?天气|今天天气怎么样)[！!。.?？,，\s]*$"
)


def classify_question(question: str) -> Decision:
    q = question.strip() if question else ""
    if SMALL_TALK_PATTERN.match(q):
        return Decision(Route.SMALL_TALK, "GREETING_RULE")
    if UNSUPPORTED_PATTERN.match(q):
        return Decision(Route.UNSUPPORTED, "GENERAL_ASSISTANT_RULE")
    return Decision(Route.KNOWLEDGE_QA, "DEFAULT_KNOWLEDGE_ROUTE")
