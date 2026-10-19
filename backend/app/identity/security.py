import base64
import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List
import bcrypt
from fastapi import Response

# 预先计算一个 dummy 密码哈希：即使用户不存在也执行同等级别的校验，降低通过响应耗时枚举账号的风险。
DUMMY_HASH = bcrypt.hashpw(b"no-such-user-timing-shield", bcrypt.gensalt(rounds=12)).decode("utf-8")

# 登录限流采用滑动窗口，键为 (IP, tenant, login)，值记录近期失败时间；同时约束来源、租户和账号维度。
_login_attempts: Dict[str, List[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def generate_token() -> str:
    """生成 32 字节密码学安全随机 token，并编码为无填充 URL-safe Base64，适合放入 Cookie/Header。"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def hash_token(token: str) -> bytes:
    """计算会话 token 的 32 字节 SHA-256 二进制摘要；数据库只保存摘要，不持久化可直接使用的明文 token。"""
    return hashlib.sha256(token.encode("utf-8")).digest()


def sign_csrf(data: str, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")


def issue_csrf_token(binding: str, secret: str) -> str:
    expiry_ms = int(time.time() * 1000) + 3600000
    raw = f"{expiry_ms}.{generate_token()}"
    sig = sign_csrf(f"{binding}:{raw}", secret)
    return f"{raw}.{sig}"


def verify_csrf_token(token: str, binding: str, secret: str) -> bool:
    if not token or not binding:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    expiry_str, rand, sig = parts
    try:
        expiry_ms = int(expiry_str)
    except ValueError:
        return False
    if expiry_ms < int(time.time() * 1000):
        return False
    expected_sig = sign_csrf(f"{binding}:{expiry_str}.{rand}", secret)
    return hmac.compare_digest(sig.encode("ascii"), expected_sig.encode("ascii"))


def check_rate_limit(ip: str, tenant: str, login: str) -> bool:
    """检查 60 秒滑动窗口登录限流：最多允许 5 次尝试；返回 False 表示当前请求应被限流拒绝。"""
    key = f"{ip}:{tenant}:{login}"
    now = time.time()
    with _rate_limit_lock:
        timestamps = _login_attempts[key]
        # 每次判断前只保留最近 60 秒的尝试记录；更早的数据退出窗口，避免列表无限增长。
        _login_attempts[key] = [t for t in timestamps if now - t <= 60.0]
        if len(_login_attempts[key]) >= 5:
            return False
        _login_attempts[key].append(now)
        return True


def set_cookie(
    response: Response,
    name: str,
    value: str,
    http_only: bool,
    secure: bool,
    max_age: int,
):
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        path="/",
        secure=secure,
        httponly=http_only,
        samesite="lax",
    )
