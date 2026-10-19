import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Request, Response, Depends
from pydantic import BaseModel
from sqlalchemy import select, and_, update
from sqlalchemy.sql import func

from app.config import settings
from app.db import get_db_connection
from app.errors import ApiException
from app.models import tenant_table, app_user_table, auth_session_table, user_group_table, group_member_table
from app.identity.security import (
    generate_token,
    hash_token,
    issue_csrf_token,
    verify_password,
    check_rate_limit,
    set_cookie,
    DUMMY_HASH,
)
from app.identity.deps import Actor, get_current_actor

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    tenantCode: str
    loginName: str
    password: str


@router.get("/csrf")
def get_csrf(request: Request, response: Response):
    session_token = request.cookies.get("RAG_SESSION")
    binding = session_token
    if not binding:
        nonce = request.cookies.get("RAG_NONCE")
        if not nonce:
            nonce = generate_token()
            set_cookie(response, "RAG_NONCE", nonce, http_only=True, secure=settings.RAG_COOKIE_SECURE, max_age=3600)
        binding = nonce

    csrf_token = issue_csrf_token(binding, settings.RAG_CSRF_SECRET)
    set_cookie(response, "RAG_CSRF", csrf_token, http_only=False, secure=settings.RAG_COOKIE_SECURE, max_age=3600)
    return {"csrfToken": csrf_token}


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response):
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not check_rate_limit(client_ip, payload.tenantCode, payload.loginName):
        raise ApiException(429, "RATE_LIMITED", "登录尝试过于频繁")

    with get_db_connection() as conn:
        stmt = (
            select(
                app_user_table.c.tenant_id,
                app_user_table.c.id,
                app_user_table.c.login_name,
                app_user_table.c.password_hash,
                app_user_table.c.is_admin,
            )
            .select_from(
                app_user_table.join(tenant_table, tenant_table.c.id == app_user_table.c.tenant_id)
            )
            .where(
                and_(
                    tenant_table.c.code == payload.tenantCode,
                    app_user_table.c.login_name == payload.loginName,
                    app_user_table.c.status == "ACTIVE",
                )
            )
        )
        user = conn.execute(stmt).first()

        if not user:
            verify_password(payload.password, DUMMY_HASH)
            raise ApiException(401, "INVALID_CREDENTIALS", "账号或密码错误")

        if not verify_password(payload.password, user.password_hash):
            raise ApiException(401, "INVALID_CREDENTIALS", "账号或密码错误")

        session_token = generate_token()
        token_h = hash_token(session_token)
        session_id = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=28800)

        now_utc = datetime.now(timezone.utc)
        conn.execute(
            auth_session_table.insert().values(
                tenant_id=user.tenant_id,
                id=session_id,
                user_id=user.id,
                token_hash=token_h,
                expires_at=expires_at,
                revoked_at=None,
                created_at=now_utc,
                updated_at=now_utc,
            )
        )
        conn.commit()

        set_cookie(response, "RAG_SESSION", session_token, http_only=True, secure=settings.RAG_COOKIE_SECURE, max_age=28800)
        set_cookie(response, "RAG_CSRF", "", http_only=False, secure=settings.RAG_COOKIE_SECURE, max_age=0)
        return {"authenticated": True}


@router.post("/logout")
def logout(request: Request, response: Response, actor: Actor = Depends(get_current_actor)):
    session_token = request.cookies.get("RAG_SESSION", "")
    token_h = hash_token(session_token)

    with get_db_connection() as conn:
        conn.execute(
            update(auth_session_table)
            .where(
                and_(
                    auth_session_table.c.tenant_id == actor.tenant_id,
                    auth_session_table.c.token_hash == token_h,
                )
            )
            .values(
                revoked_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        conn.commit()

    set_cookie(response, "RAG_SESSION", "", http_only=True, secure=settings.RAG_COOKIE_SECURE, max_age=0)
    return {"authenticated": False}
