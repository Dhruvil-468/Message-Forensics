"""
Signed-cookie session auth for the Hybrid Spam Detector app.

Sessions are a signed, tamper-proof cookie (itsdangerous) containing the
user id — no server-side session store needed. The signing key is
generated once and persisted to secret.key next to this file, so sessions
survive server restarts.
"""

import secrets
import tempfile
import os
from pathlib import Path
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, HTTPException, status

def _get_secret_key_path() -> Path:
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return Path(tempfile.gettempdir()) / "secret.key"
    try:
        test_file = Path(__file__).parent / ".write_test_auth"
        test_file.touch()
        test_file.unlink()
        return Path(__file__).parent / "secret.key"
    except Exception:
        return Path(tempfile.gettempdir()) / "secret.key"

SECRET_KEY_PATH = _get_secret_key_path()
COOKIE_NAME = "spam_forensics_session"
MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _load_or_create_secret() -> str:
    env_secret = os.environ.get("SECRET_KEY")
    if env_secret:
        return env_secret.strip()
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_text().strip()
    key = secrets.token_hex(32)
    try:
        SECRET_KEY_PATH.write_text(key)
    except Exception:
        pass
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
