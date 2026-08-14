"""
Signed-cookie session auth for the Hybrid Spam Detector app.

Sessions are a signed, tamper-proof cookie (itsdangerous) containing the
user id — no server-side session store needed. The signing key is
generated once and persisted to secret.key next to this file, so sessions
survive server restarts.
"""

import secrets
from pathlib import Path
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, HTTPException, status

SECRET_KEY_PATH = Path(__file__).parent / "secret.key"
COOKIE_NAME = "spam_forensics_session"
MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _load_or_create_secret() -> str:
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_text().strip()
    key = secrets.token_hex(32)
    SECRET_KEY_PATH.write_text(key)
    return key


_serializer = URLSafeTimedSerializer(_load_or_create_secret(), salt="session")


def create_session_token(user_id: int) -> str:
    return _serializer.dumps({"user_id": user_id})


def read_session_token(token: str):
    try:
        data = _serializer.loads(token, max_age=MAX_AGE_SECONDS)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


def get_current_user_id(request: Request) -> int:
    """FastAPI dependency: returns the logged-in user's id or raises 401."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in.")
    user_id = read_session_token(token)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired.")
    return user_id


def get_optional_user_id(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return read_session_token(token)
