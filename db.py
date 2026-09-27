"""
SQLite persistence for the Hybrid Spam Detector app.

Two tables:
  users    — account credentials (hashed, salted passwords)
  history  — every analysis a logged-in user has run, kept as their
             personal "evidence log"

Local file database: spam_forensics.db (created next to this file).
"""

import sqlite3
import json
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
import tempfile

def _get_db_path() -> Path:

    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return Path(tempfile.gettempdir()) / "spam_forensics.db"
    try:
        test_file = Path(__file__).parent / ".write_test_db"
        test_file.touch()
        test_file.unlink()
        return Path(__file__).parent / "spam_forensics.db"
    except Exception:
        return Path(tempfile.gettempdir()) / "spam_forensics.db"

DB_PATH = _get_db_path()


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            text_model_score REAL NOT NULL,
            tactic_score REAL NOT NULL,
            tactics_json TEXT NOT NULL,
            matches_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT DEFAULT 'text'
        );
        """
    )
    conn.commit()
    # Additive column migration for existing databases
    try:
        conn.execute("ALTER TABLE history ADD COLUMN source TEXT DEFAULT 'text'")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()


# ---------------------------------------------------------------------------
# Password hashing (stdlib only — PBKDF2-HMAC-SHA256, 260k iterations)
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return digest.hex(), salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(username: str, password: str):
    conn = get_conn()
    try:
        pw_hash, salt = hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, pw_hash, salt, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def authenticate(username: str, password: str):
    user = get_user_by_username(username)
    if user is None:
        return None
    if not verify_password(password, user["salt"], user["password_hash"]):
        return None
    return user


# ---------------------------------------------------------------------------
# History ("evidence log")
# ---------------------------------------------------------------------------

def add_history_entry(user_id: int, analysis: dict):
    source = analysis.get("source", "text")
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO history
           (user_id, text, verdict, confidence, text_model_score, tactic_score,
            tactics_json, matches_json, created_at, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            analysis["text"],
            analysis["verdict"],
            analysis["confidence"],
            analysis["text_model_score"],
            analysis["tactic_score"],
            json.dumps(analysis["tactics"]),
            json.dumps(analysis["matches"]),
            datetime.now(timezone.utc).isoformat(),
            source,
        ),
    )
    conn.commit()
    entry_id = cur.lastrowid
    conn.close()
    return entry_id


def list_history(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, text, verdict, confidence, created_at, source FROM history "
        "WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d.setdefault("source", "text")
        out.append(d)
    return out


def get_history_entry(user_id: int, entry_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["tactics"] = json.loads(d.pop("tactics_json"))
    d["matches"] = json.loads(d.pop("matches_json"))
    d.setdefault("source", "text")
    return d


def delete_history_entry(user_id: int, entry_id: int):
    conn = get_conn()
    cur = conn.execute(
        "DELETE FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted > 0


def export_history(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM history WHERE user_id = ? ORDER BY created_at ASC", (user_id,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["tactics"] = json.loads(d.pop("tactics_json"))
        d["matches"] = json.loads(d.pop("matches_json"))
        d.setdefault("source", "text")
        out.append(d)
    return out
