from __future__ import annotations

from __future__ import annotations
import os
import secrets
import httpx
from datetime import datetime, timedelta, UTC
from typing import Optional

import requests
from fastapi import Cookie, HTTPException, Request, Response
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from passlib.context import CryptContext
from api.db import fetch_one, execute, execute_returning
from fastapi import HTTPException
import smtplib
from email.message import EmailMessage
from disposable_email_domains import blocklist as disposable_domains

CF_TURNSTILE_SECRET = os.getenv("CF_TURNSTILE_SECRET_KEY", "")
REGISTRATION_LIMIT = 3
REGISTRATION_WINDOW_HOURS = 24


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SESSION_COOKIE_NAME = "session_token"
SESSION_DAYS = 30
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
RESET_TOKEN_HOURS = 1
VERIFY_TOKEN_HOURS = 24



def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_session(user_id: int, request: Request, response: Response) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(days=SESSION_DAYS)

    execute(
        """
        INSERT INTO user_sessions (user_id, session_token, expires_at, last_seen, user_agent, ip_address)
        VALUES (%s, %s, %s, NOW(), %s, %s)
        """,
        (
            user_id,
            token,
            expires_at,
            request.headers.get("user-agent"),
            request.client.host if request.client else None,
        ),
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,  # set True in production over HTTPS
        samesite="lax",
        max_age=SESSION_DAYS * 24 * 60 * 60,
        path="/",
    )
    return token


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )


def verify_google_credential(credential: str) -> dict:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")

    try:
        info = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        return info
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google credential: {e}")


def get_current_user(session_token: Optional[str]) -> dict:
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = fetch_one(
        """
        SELECT s.user_id, s.expires_at, u.id, u.user_name, u.email, u.first_name, u.last_name, u.status, u.auth_provider
        FROM user_sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.session_token = %s
            AND u.email_verified = TRUE
        """,
        (session_token,),
    )

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    expires_at = session["expires_at"]
    if expires_at.tzinfo is None:
        now = datetime.utcnow()
    else:
        now = datetime.now(UTC)

    if expires_at < now:
        execute("DELETE FROM user_sessions WHERE session_token = %s", (session_token,))
        raise HTTPException(status_code=401, detail="Session expired")

    if session["status"] != "active":
        raise HTTPException(status_code=403, detail="User inactive")

    execute(
        "UPDATE user_sessions SET last_seen = NOW() WHERE session_token = %s",
        (session_token,),
    )

    return {
        "id": session["id"],
        "user_name": session["user_name"],
        "email": session["email"],
        "first_name": session["first_name"],
        "last_name": session["last_name"],
        "auth_provider": session["auth_provider"],
    }


def get_current_user_id_from_cookie(session_token: Optional[str]) -> int:
    user = get_current_user(session_token)
    return user["id"]


def create_password_reset_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(hours=RESET_TOKEN_HOURS)

    execute(
        """
        INSERT INTO password_reset_tokens (user_id, token, expires_at)
        VALUES (%s, %s, %s)
        """,
        (user_id, token, expires_at),
    )
    return token

def create_email_verification_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(hours=VERIFY_TOKEN_HOURS)

    execute(
        """
        INSERT INTO email_verification_tokens (user_id, token, expires_at)
        VALUES (%s, %s, %s)
        """,
        (user_id, token, expires_at),
    )
    return token


def consume_password_reset_token(token: str) -> dict:
    row = fetch_one(
        """
        SELECT id, user_id, expires_at, used_at
        FROM password_reset_tokens
        WHERE token = %s
        """,
        (token,),
    )

    if not row:
        raise HTTPException(status_code=400, detail="Invalid reset token")

    if row["used_at"] is not None:
        raise HTTPException(status_code=400, detail="Reset token already used")

    expires_at = row["expires_at"]
    now = datetime.now(UTC) if getattr(expires_at, "tzinfo", None) else datetime.utcnow()
    if expires_at < now:
        raise HTTPException(status_code=400, detail="Reset token expired")

    execute(
        "UPDATE password_reset_tokens SET used_at = NOW() WHERE id = %s",
        (row["id"],),
    )
    return row


def consume_email_verification_token(token: str) -> dict:
    row = fetch_one(
        """
        SELECT id, user_id, expires_at, used_at
        FROM email_verification_tokens
        WHERE token = %s
        """,
        (token,),
    )

    if not row:
        return None

    if row["used_at"] is not None:
        return None

    expires_at = row["expires_at"]
    now = datetime.now(UTC) if getattr(expires_at, "tzinfo", None) else datetime.utcnow()
    if expires_at < now:
        return None

    execute(
        "UPDATE email_verification_tokens SET used_at = NOW() WHERE id = %s",
        (row["id"],),
    )
    return row


def check_disposable_email(email: str) -> None:
    domain = email.split("@")[-1].lower()
    if domain in disposable_domains or requests.get(f"https://disposable.debounce.io/?email={email}", timeout=5).json().get("disposable") == "true":
        raise HTTPException(status_code=400, detail="Disposable email addresses are not allowed")


def check_ip_registration_limit(ip: str) -> None:
    count = fetch_one(
        """
        SELECT COUNT(*) AS cnt FROM registration_attempts
        WHERE ip_address = %s AND created_at > NOW() - INTERVAL '%s hours'
        """,
        (ip, REGISTRATION_WINDOW_HOURS),
    )
    if count and count["cnt"] >= REGISTRATION_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Too many accounts registered from this IP. Try again later.",
        )


def record_registration_attempt(ip: str) -> None:
    execute(
        "INSERT INTO registration_attempts (ip_address, created_at) VALUES (%s, NOW())",
        (ip,),
    )


async def verify_turnstile_token(token: str) -> None:
    if not CF_TURNSTILE_SECRET:
        raise HTTPException(status_code=500, detail="Turnstile not configured")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": CF_TURNSTILE_SECRET, "response": token},
        )
    result = resp.json()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail="CAPTCHA verification failed")


def send_email(to_email: str, subject: str, body: str, html: str | None = None) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    if not all([smtp_host, smtp_port, smtp_user, smtp_password]):
        raise RuntimeError("SMTP config is missing")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "verbox.ai <anton@verbox.ai>"
    msg["To"] = to_email
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if use_tls:
                server.starttls()

            server.login(smtp_user, smtp_password)
            server.send_message(msg)

    except Exception as e:
        print("EMAIL SEND ERROR:", e)
        raise