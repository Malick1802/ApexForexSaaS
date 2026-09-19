"""
core/auth.py -- Portal login accounts for ApexForexSaaS.
Separate from user_accounts.py (MT5 credentials).
Linked by email.
"""
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTAL_DB_PATH = PROJECT_ROOT / "portal_users.db"

ADMIN_EMAIL    = "malicktra99@gmail.com"
ADMIN_PASSWORD = "Justin180289"
ADMIN_NAME     = "Admin"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(str(PORTAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_portal_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portal_users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'subscriber',
            created_at    TEXT    NOT NULL
        )
    """)
    conn.commit()
    existing = conn.execute(
        "SELECT id FROM portal_users WHERE email = ?", (ADMIN_EMAIL,)
    ).fetchone()
    if not existing:
        conn.execute("""
            INSERT INTO portal_users (name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, 'admin', ?)
        """, (ADMIN_NAME, ADMIN_EMAIL, _hash(ADMIN_PASSWORD), _now_iso()))
        conn.commit()
    conn.close()


def register_portal_user(name: str, email: str, password: str):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO portal_users (name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, 'subscriber', ?)
        """, (name.strip(), email.strip().lower(), _hash(password), _now_iso()))
        conn.commit()
        user = conn.execute(
            "SELECT * FROM portal_users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return dict(user) if user else None
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def verify_login(email: str, password: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM portal_users WHERE email = ? AND password_hash = ?",
        (email.strip().lower(), _hash(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_portal_user_by_email(email: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM portal_users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_portal_users():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM portal_users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_portal_user(email: str):
    conn = get_connection()
    conn.execute("DELETE FROM portal_users WHERE email = ?", (email.strip().lower(),))
    conn.commit()
    conn.close()


def update_portal_user(email: str, name: str):
    conn = get_connection()
    conn.execute(
        "UPDATE portal_users SET name = ? WHERE email = ?",
        (name.strip(), email.strip().lower())
    )
    conn.commit()
    conn.close()


def change_password(email: str, new_password: str):
    conn = get_connection()
    conn.execute(
        "UPDATE portal_users SET password_hash = ? WHERE email = ?",
        (_hash(new_password), email.strip().lower())
    )
    conn.commit()
    conn.close()


init_portal_db()
