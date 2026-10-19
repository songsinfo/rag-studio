import re
from app.errors import ApiException

PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*")
CREDENTIAL_PATTERN = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=：]\s*[^\s,;，；]{8,}")
INJECTION_PATTERN = re.compile(
    r"(?iu)^\s*(?:please\s+)?(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?|reveal\s+(?:the\s+)?system\s+prompt|(?:请)?忽略.{0,12}(?:系统|之前|以上).{0,8}(?:指令|提示)|(?:请)?(?:泄露|显示|输出).{0,8}(?:系统提示词|system\s+prompt))"
)


def contains_secret(text: str) -> bool:
    return bool(
        PRIVATE_KEY_PATTERN.search(text)
        or BEARER_PATTERN.search(text)
        or CREDENTIAL_PATTERN.search(text)
    )


def contains_pii(text: str) -> bool:
    return bool(PHONE_PATTERN.search(text) or ID_CARD_PATTERN.search(text))


def validate_input(text: str) -> None:
    if not text:
        return
    if INJECTION_PATTERN.search(text):
        raise ApiException(400, "PROMPT_INJECTION_BLOCKED", "请求包含试图修改系统指令的内容")
    if contains_secret(text):
        raise ApiException(400, "SECRET_INPUT_BLOCKED", "请求包含疑似密钥或凭据")
    if contains_pii(text):
        raise ApiException(400, "PII_INPUT_BLOCKED", "请求包含高置信度个人敏感信息")


def validate_output(text: str) -> None:
    if not text:
        return
    if contains_secret(text) or contains_pii(text):
        raise ApiException(422, "SENSITIVE_OUTPUT_BLOCKED", "回答包含高置信度敏感信息")
