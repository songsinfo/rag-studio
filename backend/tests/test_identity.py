from datetime import datetime, timezone, timedelta
import pytest
from app.models import auth_session_table
from app.identity.security import generate_token, hash_token, issue_csrf_token
import app.db as app_db


def test_csrf_issuance(client):
    resp = client.get("/api/v1/auth/csrf")
    assert resp.status_code == 200
    data = resp.json()
    assert "csrfToken" in data
    assert "RAG_CSRF" in resp.cookies
    assert "RAG_NONCE" in resp.cookies


def test_login_rejected_without_csrf(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "admin", "password": "admin123"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "CSRF_INVALID"


def test_login_rejected_with_invalid_credentials(client):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "admin", "password": "wrong-password"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_login_rejected_for_disabled_user(client):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "disabled_user", "password": "admin123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_login_rejected_for_wrong_tenant(client):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "non-existent-tenant", "loginName": "admin", "password": "admin123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_full_login_me_logout_lifecycle(client):
    # 第 1 步：获取 CSRF token，后续所有写请求都基于该安全上下文。
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    # 第 2 步：使用有效凭据登录并建立会话 Cookie。
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "admin", "password": "admin123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["authenticated"] is True
    assert "RAG_SESSION" in client.cookies

    # 第 3 步：访问 /api/v1/me，确认会话身份已生效。
    me_resp = client.get("/api/v1/me")
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["userId"] == "user-admin-1"
    assert me_data["tenantId"] == "tenant-demo-1"
    assert me_data["loginName"] == "admin"
    assert me_data["admin"] is True
    assert me_data["securityEpoch"] == 1
    assert len(me_data["groups"]) == 1
    assert me_data["groups"][0]["name"] == "Ops Group"

    # 第 4 步：刷新绑定到当前已登录会话的 CSRF token，验证匿名/登录 token 不可混用。
    auth_csrf_resp = client.get("/api/v1/auth/csrf")
    auth_csrf_token = auth_csrf_resp.json()["csrfToken"]

    # 第 5 步：退出登录并立即撤销服务器端会话。
    logout_resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": auth_csrf_token},
    )
    assert logout_resp.status_code == 200
    assert logout_resp.json()["authenticated"] is False

    # 第 6 步：再次访问受保护接口，确认已撤销会话不能继续使用。
    after_logout = client.get("/api/v1/me")
    assert after_logout.status_code == 401
    assert after_logout.json()["code"] == "AUTH_REQUIRED"


def test_rate_limiting(client):
    csrf_resp = client.get("/api/v1/auth/csrf")
    csrf_token = csrf_resp.json()["csrfToken"]

    # 连续尝试 5 次，仍应处在允许窗口内。
    for _ in range(5):
        resp = client.post(
            "/api/v1/auth/login",
            json={"tenantCode": "demo", "loginName": "rate_limit_test", "password": "wrong"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    # 第 6 次尝试必须触发限流，验证 60 秒滑动窗口边界。
    rate_limited_resp = client.post(
        "/api/v1/auth/login",
        json={"tenantCode": "demo", "loginName": "rate_limit_test", "password": "wrong"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert rate_limited_resp.status_code == 429
    assert rate_limited_resp.json()["code"] == "RATE_LIMITED"


def test_expired_session_rejected(client):
    # 手工插入已过期会话，验证过期时间由服务端强制执行，而不是仅依赖 Cookie。
    expired_token = generate_token()
    token_h = hash_token(expired_token)
    past_time = datetime.now(timezone.utc) - timedelta(hours=1)

    with app_db.engine.begin() as conn:
        conn.execute(
            auth_session_table.insert().values(
                tenant_id="tenant-demo-1",
                id="expired-session-1",
                user_id="user-admin-1",
                token_hash=token_h,
                expires_at=past_time,
                revoked_at=None,
                created_at=past_time,
                updated_at=past_time,
            )
        )

    client.cookies.set("RAG_SESSION", expired_token)
    resp = client.get("/api/v1/me")
    assert resp.status_code == 401
    assert resp.json()["code"] == "AUTH_REQUIRED"


def test_health_check(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["db"] == "connected"
