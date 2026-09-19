"""
core/sessions.py
Server-side session token store for ApexForexSaaS.
Tokens are persisted in portal_users.db and survive Streamlit WebSocket drops.
The token is kept in the browser via st.query_params so a reconnect restores auth.
"""
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTAL_DB_PATH = PROJECT_ROOT / "portal_users.db"

SESSION_TTL_HOURS = 72  # Token valid for 3 days


def _conn():
    c = sqlite3.connect(str(PORTAL_DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)).isoformat()


def init_sessions_table():
    """Create the sessions table if it doesn't exist."""
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portal_sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            email       TEXT NOT NULL,
            name        TEXT NOT NULL,
            role        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session(user: dict) -> str:
    """
    Create a new session for a logged-in user.
    Returns the session token (a 32-byte hex string).
    """
    token = secrets.token_hex(32)
    conn = _conn()
    conn.execute("""
        INSERT INTO portal_sessions (token, user_id, email, name, role, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        token,
        user["id"],
        user["email"],
        user["name"],
        user["role"],
        _now(),
        _expiry(),
    ))
    conn.commit()
    conn.close()
    return token


def validate_session(token: str) -> dict | None:
    """
    Validate a session token. Returns user dict if valid and not expired, else None.
    Automatically cleans up expired sessions.
    """
    if not token or len(token) != 64:
        return None
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM portal_sessions WHERE token = ?", (token,)
    ).fetchone()

    if not row:
        conn.close()
        return None

    # Check expiry
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            conn.execute("DELETE FROM portal_sessions WHERE token = ?", (token,))
            conn.commit()
            conn.close()
            return None
    except Exception:
        conn.close()
        return None

    conn.close()
    return {
        "id":    row["user_id"],
        "email": row["email"],
        "name":  row["name"],
        "role":  row["role"],
    }


def touch_session(token: str):
    """Extend an active session's expiry by SESSION_TTL_HOURS from now."""
    if not token or len(token) != 64:
        return
    try:
        conn = _conn()
        conn.execute(
            "UPDATE portal_sessions SET expires_at = ? WHERE token = ?",
            (_expiry(), token)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def delete_session(token: str):
    """Delete a session (logout)."""
    if not token:
        return
    conn = _conn()
    conn.execute("DELETE FROM portal_sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def purge_expired():
    """Remove all expired sessions. Call periodically."""
    conn = _conn()
    conn.execute("DELETE FROM portal_sessions WHERE expires_at < ?", (_now(),))
    conn.commit()
    conn.close()


# Auto-init on import
init_sessions_table()

